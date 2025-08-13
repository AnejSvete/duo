import collections
import copy
import os
import pickle

import fsspec
import numpy as np
import torch
import torch.nn.functional as F

import trainer_base
import utils


class AR(trainer_base.TrainerBase):
    def generate_conditioned(self, prompts, mode="random", top_k=1):
        """
        Generate completions conditioned on prompts, using the specified unmasking mode.
        prompts: (batch, seq) tensor (padded)
        Returns: (batch, seq) tensor (same shape as targets)
        """
        # Find the length of the prompt for each sample (first pad after non-pad)
        pad = self.tokenizer.pad_token_id
        batch_size, seq_len = prompts.shape
        prompt_lens = (prompts != pad).sum(dim=1)
        # We'll generate up to max_target_len tokens for each sample
        # For simplicity, assume all targets are same length as in validation
        # (the caller should pad appropriately)
        max_gen_len = seq_len
        device = prompts.device
        x = prompts.clone()
        # For AR, generate one token at a time after the prompt
        for t in range(seq_len):
            # Only fill positions after the prompt
            mask = (t >= prompt_lens) & (t < max_gen_len)
            if not mask.any():
                continue
            # Prepare input up to t for each sample
            input_toks = []
            for i in range(batch_size):
                end = t if t < max_gen_len else max_gen_len
                input_toks.append(x[i, :end])
            # Pad to same length
            max_inp = max([it.size(0) for it in input_toks])
            input_batch = torch.stack(
                [F.pad(it, (0, max_inp - it.size(0)), value=pad) for it in input_toks]
            )
            # Get logits for next token
            with torch.no_grad():
                logits = self.backbone(input_batch, None)  # (B, L, V)
                logits = logits[:, -1, :]  # (B, V)
                logits[:, self.mask_index] = self.neg_infinity
                probs = logits.softmax(-1)
            # Choose next token according to mode
            if mode == "random":
                next_tok = torch.multinomial(probs, 1).squeeze(-1)
            elif mode == "one_level":
                # For AR, treat as greedy (since only one token at a time)
                next_tok = probs.argmax(-1)
            elif mode == "top_k":
                # For AR, top_k is greedy for k=1, else sample from top-k
                if top_k == 1:
                    next_tok = probs.argmax(-1)
                else:
                    topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
                    topk_probs = topk_probs / topk_probs.sum(-1, keepdim=True)
                    sampled = torch.multinomial(topk_probs, 1).squeeze(-1)
                    next_tok = topk_idx.gather(1, sampled.unsqueeze(-1)).squeeze(-1)
            else:
                raise ValueError(f"Unknown generation mode: {mode}")
            # Fill in next token for those still generating
            for i in range(batch_size):
                if mask[i]:
                    x[i, t] = next_tok[i]
        # Return only the generated part (after prompt)
        # But for validation, we want the full sequence (to compare to targets)
        return x[:, :seq_len]

    def __init__(self, config, tokenizer):
        vocab_size = tokenizer.vocab_size
        if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert not self.config.algo.time_conditioning
        assert self.config.prior.type == "none"

    def _process_model_input(self, x0, valid_tokens):
        input_tokens = x0[:, :-1]
        output_tokens = x0[:, 1:]
        valid_tokens = valid_tokens[:, 1:]
        return input_tokens, output_tokens, valid_tokens

    def nll(self, input_tokens, output_tokens, current_accumulation_step):
        del current_accumulation_step
        output = self.backbone(input_tokens, None)
        output[:, :, self.mask_index] = self.neg_infinity
        output = output.log_softmax(-1)
        return -output.gather(-1, output_tokens[:, :, None])[:, :, 0]

    def generate_samples(self, num_samples, **kwargs):
        # precompute token buffer
        num_pred_tokens = self.num_tokens - 1
        x = torch.zeros(
            (num_samples, num_pred_tokens + 1), dtype=torch.long, device=self.device
        )
        x[:, 0] = self.tokenizer.bos_token_id
        # precompute noise
        noise = (
            torch.distributions.Gumbel(0, 1)
            .sample((num_samples, num_pred_tokens, self.vocab_size))
            .to(self.device)
        )
        if self.config.sampling.use_float64:
            noise = noise.to(torch.float64)
        for i in range(num_pred_tokens):
            output = self.backbone(x[:, : i + 1], None)
            output[:, :, self.mask_index] = self.neg_infinity
            output = output.log_softmax(-1)
            y = (output[:, -1, :] + noise[:, i, :]).argmax(-1)
            x[:, i + 1] = y
        return x

    def _process_sigma(self, sigma):
        del sigma
        return None


class MDLM(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        # ancestral sampling isn't desirable because it's slow
        assert self.sampler == "ancestral_cache"

    # def _process_model_output(self, model_output, xt, sigma):
    #   del sigma
    #   model_output[:, :, self.mask_index] += self.neg_infinity

    #   # Normalize the model_output such that x.exp() is
    #   # a probability distribution over vocab_size.
    #   model_output = model_output - torch.logsumexp(
    #     model_output, dim=-1, keepdim=True)
    #   # Apply updates directly in the logits matrix.
    #   # For the logits of the unmasked tokens, set all values
    #   # to -infinity except for the indices corresponding to
    #   # the unmasked tokens.
    #   unmasked_indices = (xt != self.mask_index)
    #   model_output[unmasked_indices] = self.neg_infinity
    #   model_output[unmasked_indices, xt[unmasked_indices]] = 0
    #   return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        del xt
        log_p_theta = torch.gather(
            input=log_x_theta, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)
        return log_p_theta * dalpha_t / (1 - alpha_t)

    def _get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        # score(x, t) = p_t(y) / p_t(x)
        # => log score(x, t) = log p_t(y) - log p_t(x)

        # case 1: x = masked
        #   (i) y = unmasked
        #     log score(x, t) = log p_\theta(x)|_y + log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))
        #   (ii) y = masked
        #     log score(x, t) = 0

        # case 2: x = unmasked
        #   (i) y != masked, y != x
        #     log score(x_i, t) = - inf
        #   (ii) y = x
        #     log score(x_i, t) = 0
        #   (iii) y = masked token
        #     log score(x_i, t) = - log k
        #     where k = exp(- sigma) / (1 - exp(- sigma))

        log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
        assert log_k.ndim == 1

        masked_score = model_output + log_k[:, None, None]
        masked_score[:, :, self.mask_index] = 0

        unmasked_score = self.neg_infinity * torch.ones_like(model_output)
        unmasked_score = torch.scatter(
            unmasked_score, -1, x[..., None], torch.zeros_like(unmasked_score[..., :1])
        )
        unmasked_score[:, :, self.mask_index] = -(log_k[:, None] * torch.ones_like(x))

        masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
        model_output = masked_score * masked_indices + unmasked_score * (
            1 - masked_indices
        )
        return model_output.exp()

    def _process_model_output(self, model_output, xt, sigma):
        # For MDLM, mask out the mask token and normalize
        del sigma
        model_output[:, :, self.mask_index] += self.neg_infinity
        # Normalize to log-probabilities
        model_output = model_output - torch.logsumexp(
            model_output, dim=-1, keepdim=True
        )
        # Optionally, mask out unmasked positions (if needed for your loss)
        unmasked_indices = xt != self.mask_index
        model_output[unmasked_indices] = self.neg_infinity
        model_output[unmasked_indices, xt[unmasked_indices]] = 0
        return model_output

    def generate_conditioned(self, prompts, mode="random", top_k=1):
        """
        Generate completions conditioned on prompts, using the specified unmasking mode.
        prompts: (batch, seq) tensor (padded)
        Returns: (batch, seq) tensor (same shape as targets)
        """
        pad = self.tokenizer.pad_token_id
        mask_token = self.mask_index
        batch_size, seq_len = prompts.shape
        # Find the length of the prompt for each sample (first pad after non-pad)
        prompt_lens = (prompts != pad).sum(dim=1)
        # Start with prompt, mask the rest
        x = prompts.clone()
        for i in range(batch_size):
            x[i, prompt_lens[i] :] = mask_token
        finished = torch.zeros(batch_size, dtype=torch.bool, device=prompts.device)
        for step in range(seq_len):
            mask_pos = x == mask_token
            if not mask_pos.any():
                break
            # Use sigma=0 for all positions (default denoising step)
            sigma = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            with torch.no_grad():
                logits = self.backbone(x, sigma)  # (B, L, V)
                logits[:, :, mask_token] = self.neg_infinity
                probs = logits.softmax(-1)
            for i in range(batch_size):
                if finished[i]:
                    continue
                masked_indices = mask_pos[i].nonzero(as_tuple=True)[0]
                if len(masked_indices) == 0:
                    finished[i] = True
                    continue
                if mode == "random":
                    pos = masked_indices[
                        torch.randint(0, len(masked_indices), (1,))
                    ].item()
                    token = torch.multinomial(probs[i, pos], 1).item()
                    x[i, pos] = token
                elif mode == "one_level":
                    start = prompt_lens[i].item()
                    ids = x[i]
                    next_bar = (
                        ids[start:] == self.tokenizer.convert_tokens_to_ids("|")
                    ).nonzero(as_tuple=True)
                    if len(next_bar[0]) > 0:
                        end = start + next_bar[0][0].item()
                    else:
                        end = seq_len
                    for pos in range(start, end):
                        if x[i, pos] == mask_token:
                            token = probs[i, pos].argmax().item()
                            x[i, pos] = token
                    prompt_lens[i] = end + 1
                elif mode == "top_k":
                    masked_probs = probs[i][mask_pos[i]]
                    if masked_probs.size(0) == 0:
                        finished[i] = True
                        continue
                    conf, idx = masked_probs.max(dim=1)
                    k = min(top_k, len(conf))
                    topk_idx = conf.topk(k).indices
                    masked_positions = mask_pos[i].nonzero(as_tuple=True)[0]
                    for j in topk_idx:
                        pos = masked_positions[j.item()].item()
                        token = probs[i, pos].argmax().item()
                        x[i, pos] = token
                else:
                    raise ValueError(f"Unknown generation mode: {mode}")
        for step in range(seq_len):
            mask_pos = x == mask_token
            if not mask_pos.any():
                break
            # Use sigma=0 for all positions (final denoising step)
            sigma = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            with torch.no_grad():
                logits = self.backbone(x, sigma)  # (B, L, V)
                logits[:, :, mask_token] = self.neg_infinity
                probs = logits.softmax(-1)
            for i in range(batch_size):
                if finished[i]:
                    continue
                masked_indices = mask_pos[i].nonzero(as_tuple=True)[0]
                if len(masked_indices) == 0:
                    finished[i] = True
                    continue
                if mode == "random":
                    pos = masked_indices[
                        torch.randint(0, len(masked_indices), (1,))
                    ].item()
                    token = torch.multinomial(probs[i, pos], 1).item()
                    x[i, pos] = token
                elif mode == "one_level":
                    start = prompt_lens[i].item()
                    ids = x[i]
                    next_bar = (
                        ids[start:] == self.tokenizer.convert_tokens_to_ids("|")
                    ).nonzero(as_tuple=True)
                    if len(next_bar[0]) > 0:
                        end = start + next_bar[0][0].item()
                    else:
                        end = seq_len
                    for pos in range(start, end):
                        if x[i, pos] == mask_token:
                            token = probs[i, pos].argmax().item()
                            x[i, pos] = token
                    prompt_lens[i] = end + 1
                elif mode == "top_k":
                    masked_probs = probs[i][mask_pos[i]]
                    if masked_probs.size(0) == 0:
                        finished[i] = True
                        continue
                    conf, idx = masked_probs.max(dim=1)
                    k = min(top_k, len(conf))
                    topk_idx = conf.topk(k).indices
                    masked_positions = mask_pos[i].nonzero(as_tuple=True)[0]
                    for j in topk_idx:
                        pos = masked_positions[j.item()].item()
                        token = probs[i, pos].argmax().item()
                        x[i, pos] = token
                else:
                    raise ValueError(f"Unknown generation mode: {mode}")
        return x[:, :seq_len]


class D3PMAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.noise.type == "log-linear"
        assert self.parameterization == "mean"

    def _process_model_output(self, model_output, xt, sigma):
        del xt
        del sigma
        if self.subs_masking:
            model_output[:, :, self.mask_index] += self.neg_infinity
        return model_output.log_softmax(dim=-1)

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        del dalpha_t
        assert not low_var
        dt = 1 / self.T
        t = 1 - alpha_t  # Only valid for log-linear schedule.
        t = t.clamp(0.0, 1.0 - 1e-4)
        alpha_t = alpha_t + torch.zeros_like(xt)
        alpha_s = t - dt + torch.zeros_like(xt)
        assert alpha_s.shape == xt.shape
        assert alpha_t.shape == xt.shape
        log_x_theta_at_x0 = torch.gather(log_x_theta, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = log_x_theta[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)
        L_vb_masked = term_1_coef * (term_1_log_nr - term_1_log_dr) + term_2_coef * (
            term_2_log_nr - term_2_log_dr
        )

        diffusion_loss = self.T * L_vb_masked * (xt == self.mask_index)
        return self._reconstruction_loss(x0) + diffusion_loss


class SEDDAbsorb(trainer_base.AbsorbingState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.predictor == "analytic"

    def _get_score(self, x, sigma):
        return self.forward(x, sigma).exp()

    def _process_model_output(self, model_output, xt, sigma):
        esigm1_log = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(model_output.dtype)
        )
        # logits shape
        # (batch_size, context_length, vocab_size)
        model_output = (
            model_output
            - esigm1_log[:, None, None]
            - np.log(model_output.shape[-1] - 1)
        )
        # The below scatter operation sets the log score
        # for the input word to 0.
        model_output = torch.scatter(
            model_output, -1, xt[..., None], torch.zeros_like(model_output[..., :1])
        )
        return model_output

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        """Computes the SEDD loss for the Absorbing State Diffusion.

        Args:
          log_x_theta: float torch.Tensor with shape (batch_size,
              context_length, vocab_size),
              log score, output of the denoising network.
          xt: int torch.Tensor with shape (batch_size,
              context_length), input.
          x0: int torch.Tensor with shape (batch_size,
              context_length), input.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          alpha_t: float torch.Tensor with shape (batch_size, 1),
              signal level.
          dalpha_t: float or float torch.Tensor with shape (batch_size, 1),
              time derivative of signal level.
          low_var: bool, low variance loss during training.

        Returns:
          loss with shape (batch_size, context_length).
        """
        assert not low_var
        masked_indices = xt == self.mask_index
        sigma = self._sigma_from_alphat(alpha_t)
        dsigma = -dalpha_t / alpha_t

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_x_theta[masked_indices], -1, words_that_were_masked[..., None]
        ).squeeze(-1)
        score = log_x_theta[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(dim=-1) + score[
                :, self.mask_index + 1 :
            ].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(*xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return dsigma * entropy


class DUO_BASE(trainer_base.UniformState):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self._validate_configuration()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = collections.OrderedDict(
            (k, v)
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("teacher")
        )
        super().on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = collections.OrderedDict(
            (k, v)
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("teacher")
        )
        super().on_load_checkpoint(checkpoint)

    def _process_model_output(self, model_output, xt, sigma):
        del xt, sigma
        return model_output.log_softmax(dim=-1)

    def _compute_posterior(self, x, xt, alpha_s, alpha_t):
        """Computes the posterior / approximate posterior.

        Args:
          x: Either clean input `x0` (one-hot),
            or model's predicted `x_theta` of shape (B, L, V).
          xt: The noisy latent (as indices) of shape (B, L).
          alpha_s: Noise level at s of shape (B, [L | 1], 1).
          alpha_t: Noise level at t of shape (B, [L | 1], 1).

        Returns:
          Posterior / approximate posterior of shape (B, L, V).
        """
        if self.config.sampling.use_float64:
            x = x.to(torch.float64)
        if alpha_s.ndim == 2:
            alpha_s = alpha_s.unsqueeze(-1)
        if alpha_t.ndim == 2:
            alpha_t = alpha_t.unsqueeze(-1)
        alpha_ts = alpha_t / alpha_s
        d_alpha = alpha_s - alpha_t
        xt_one_hot = F.one_hot(xt, self.vocab_size).to(self.dtype).to(self.device)
        return (
            alpha_t * self.vocab_size * x * xt_one_hot
            + (alpha_ts - alpha_t) * xt_one_hot
            + d_alpha * x
            + (1 - alpha_ts) * (1 - alpha_s) / self.vocab_size
        ) / (
            alpha_t * self.vocab_size * torch.gather(x, -1, xt[..., None])
            + (1 - alpha_t)
        )

    def nll_per_token(self, log_x_theta, xt, x0, alpha_t, dalpha_t, low_var=False):
        assert alpha_t.ndim == 2
        assert x0.ndim == 2
        assert xt.ndim == 2
        assert not torch.is_tensor(dalpha_t) or dalpha_t.ndim == 2
        x_reconst = log_x_theta.exp()
        x_bar_theta = (
            self.vocab_size * alpha_t[:, :, None] * x_reconst + 1 - alpha_t[:, :, None]
        )
        coeff = dalpha_t / (self.vocab_size * alpha_t)
        x_eq_xt = (x0 == xt).float()
        x_neq_xt = 1 - x_eq_xt
        xbar_xt = (1 - alpha_t) + self.vocab_size * alpha_t * x_eq_xt
        xbar_theta_xt = torch.gather(x_bar_theta, -1, xt.unsqueeze(-1)).squeeze(-1)
        xbar_theta_x = torch.gather(x_bar_theta, -1, x0.unsqueeze(-1)).squeeze(-1)
        term1 = self.vocab_size * (1 / xbar_xt - 1 / xbar_theta_xt)

        const = (1 - alpha_t) / (self.vocab_size * alpha_t + 1 - alpha_t)
        term2_coefs = x_eq_xt * const + x_neq_xt
        term2_offset = (
            (self.vocab_size - 1) * const * x_eq_xt - (1 / const) * x_neq_xt
        ) * const.log()
        term2_theta = -term2_coefs * (
            x_bar_theta.log().sum(-1) - self.vocab_size * xbar_theta_xt.log()
        )
        term2_theta = (
            term2_theta
            - self.vocab_size
            * alpha_t
            / (1 - alpha_t)
            * (xbar_theta_x.log() - xbar_theta_xt.log())
            * x_neq_xt
        )
        term2 = term2_theta + term2_offset
        diffusion_loss = coeff * (term1 - term2)
        assert diffusion_loss.ndim == 2
        return diffusion_loss

    def _ancestral_update(self, x, t, dt, p_x0=None, noise_removal_step=False):
        del p_x0
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        sigma_t = self._sigma_from_alphat(alpha_t)
        assert alpha_t.ndim == 2

        q_xs = self._compute_posterior(
            x=self.forward(x, sigma_t).exp(), xt=x, alpha_s=alpha_s, alpha_t=alpha_t
        )
        if self.p_nucleus < 1:
            q_xs = utils.top_k_top_p_filtering(q_xs.log(), top_p=self.p_nucleus)
        return None, trainer_base.sample_categorical(q_xs)


class Integral(torch.autograd.Function):
    """
    torch module calculating UDLM's p_t
    """

    @staticmethod
    def forward(ctx, gamma_t, data):
        gamma_max = data["gamma_max"]
        gamma_min = data["gamma_min"]
        if (gamma_t.max() > gamma_max) or (gamma_t.min() < gamma_min):
            print("max:{} {}".format(gamma_t.max(), gamma_max))
            print("min:{} {}".format(gamma_t.min(), gamma_min))
            gamma_t = torch.clip(gamma_t, gamma_min, gamma_max)
        indices = torch.round(
            (data["num_points"] - 1) * (gamma_t - gamma_min) / (gamma_max - gamma_min)
        ).long()
        grad_pt = data["grad_pt"]
        ctx.grad_pt = grad_pt[indices]

        pt = data["pt"][indices]
        assert pt.shape == gamma_t.shape
        return pt

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.grad_pt * grad_output, None


class DUO(DUO_BASE):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        with fsspec.open(self.config.algo.integral_cache_path, "rb") as f:
            self.integral_cache = pickle.load(f)
        self.integral_cache["pt"] = torch.from_numpy(self.integral_cache["pt"])
        self.integral_cache["grad_pt"] = torch.from_numpy(
            self.integral_cache["grad_pt"]
        )
        self.gamma_min = self.config.algo.gamma_min
        self.gamma_max = self.config.algo.gamma_max
        self.gumbel_tau_log10_start = self.config.algo.gumbel_tau_log10_start
        self.gumbel_tau_log10_end = self.config.algo.gumbel_tau_log10_end
        self.curriculum_start = self.config.algo.curriculum_start
        self.curriculum_end = self.config.algo.curriculum_end
        self.loss_type = self.config.algo.loss_type
        self._validate_configuration()

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.integral_cache["pt"] = self.integral_cache["pt"].to(*args, **kwargs)
        self.integral_cache["grad_pt"] = self.integral_cache["grad_pt"].to(
            *args, **kwargs
        )
        return self

    def _compute_gumbel_tau_inverse(self):
        start = self.gumbel_tau_log10_start
        end = self.gumbel_tau_log10_end
        delta = end - start
        if self.global_step < self.curriculum_start:
            tau = start
        elif self.global_step < self.curriculum_end:
            frac = (self.global_step - self.curriculum_start) / (
                self.curriculum_end - self.curriculum_start
            )
            tau = start + frac * delta
        else:
            tau = -10
        return 10 ** (-tau)

    def training_step(self, batch, batch_idx):
        self.log(
            name="gumbel_tau_log10",
            value=1 / self._compute_gumbel_tau_inverse(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return super().training_step(batch, batch_idx)

    def _gamma_to_alphat(self, gamma_t):
        integral = Integral.apply(gamma_t, self.integral_cache)
        return (self.vocab_size * integral - 1) / (self.vocab_size - 1)

    def _prior_loss(self):
        alpha_1 = self._gamma_to_alphat(torch.tensor(self.gamma_max))
        loss = (alpha_1 + (1 - alpha_1) / self.vocab_size) * torch.log(
            (self.vocab_size - 1) * alpha_1 + 1
        ) + (1 - 1 / self.vocab_size) * (1 - alpha_1) * torch.log(1 - alpha_1)
        return loss.item()

    def _q_xt_gaussian(self, x, gamma_t):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert x.ndim == 3
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()
        epsilon = torch.randn(x.shape, dtype=torch.float32, device=self.device)
        return alpha_t * x + sigma_t * epsilon

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=False):
        use_true_nll = self.global_step > self.curriculum_end or not train_mode
        if use_true_nll:
            return super().nll(x0, output_tokens, current_accumulation_step)
        del output_tokens
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
        gamma_t_prime = self.gamma_max - self.gamma_min
        usdm_alpha_t = self._gamma_to_alphat(gamma_t)
        T = 1000
        usdm_dalpha_t = (
            gamma_t_prime * T * (self._gamma_to_alphat(gamma_t + 1 / T) - usdm_alpha_t)
        )
        usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
        usdm_dalpha_t = usdm_dalpha_t.unsqueeze(-1)
        assert usdm_alpha_t.ndim == 2
        sigma = self._sigma_from_alphat(usdm_alpha_t)

        x0_one_hot = F.one_hot(x0, self.vocab_size)
        xt = self._q_xt_gaussian(x0_one_hot, gamma_t)
        xt = xt * self._compute_gumbel_tau_inverse()
        xt_usdm = xt.argmax(-1)
        log_x_theta = self.forward(xt, sigma=sigma)

        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt_usdm,
            x0=x0,
            alpha_t=usdm_alpha_t,
            dalpha_t=usdm_dalpha_t,
            low_var=False,
        )


class Distillation(DUO):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.update_teacher_every = config.algo.update_teacher_every
        self.save_hyperparameters()
        self.teacher = None
        self.teacher_ema = config.algo.teacher_ema
        self.linear_growth_dt = config.algo.linear_growth_dt
        self.linear_growth_min = config.algo.linear_growth_min
        self.linear_growth_max = config.algo.linear_growth_max

    def _validate_configuration(self):
        assert os.path.exists(self.config.algo.integral_cache_path), (
            "The integral cache (Eq. 10 in the paper) for "
            f"the {self.config.data.tokenizer_name_or_path} "
            " tokenizer doesnt exist at "
            f"{self.config.algo.integral_cache_path}. "
            "Please generate it by running the utils.py script, "
            "and ensure the correct path is specified using the "
            "algo.integral_cache_path flag."
        )
        assert self.loss_type in {"kl-fwd", "kl-bwd", "posterior", "kl-posterior"}

    def _maybe_update_teacher_weights(self):
        if self.global_step % self.update_teacher_every != 0:
            return
        if self.teacher_ema:
            self.ema.copy_to(self.teacher.parameters())
        else:
            for better_param, current_param in zip(
                self.backbone.parameters(), self.teacher.parameters()
            ):
                if current_param.requires_grad:
                    current_param.data.copy_(better_param.data)

    @torch.no_grad()
    def _teacher_logits(self, xt, sigma):
        if self.teacher is None:
            self.teacher = copy.deepcopy(self.backbone)
        self._maybe_update_teacher_weights()

        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.teacher(xt, sigma)
        logits = self._process_model_output(
            model_output=model_output, xt=xt, sigma=sigma
        )
        return logits.detach()

    def _sample_trajectory(self, x0, gamma_t, gamma_s):
        """Computes the noisy sample xt."""
        assert gamma_t.ndim == 1
        assert gamma_s.ndim == 1
        assert x0.ndim == 2
        x0 = F.one_hot(x0, self.vocab_size).to(self.dtype).to(self.device)
        gamma_t = gamma_t.unsqueeze(-1).unsqueeze(-1)
        alpha_t = torch.sigmoid(-gamma_t).sqrt()
        sigma_t = torch.sigmoid(gamma_t).sqrt()

        gamma_s = gamma_s.unsqueeze(-1).unsqueeze(-1)
        alpha_s = torch.sigmoid(-gamma_s).sqrt()
        sigma_s = torch.sigmoid(gamma_s).sqrt()

        epsilon = torch.randn(x0.shape, dtype=torch.float32, device=self.device)
        xt = alpha_t * x0 + sigma_t * epsilon
        xs = alpha_s * x0 + sigma_s * epsilon
        return xt, xs

    def _compute_dt(self):
        if self.linear_growth_dt:
            scale = self.global_step / self.trainer.max_steps
            return self.linear_growth_min + scale * (
                self.linear_growth_max - self.linear_growth_min
            )
        n = self.global_step // self.update_teacher_every
        return 2**n / self.T

    def nll(self, x0, output_tokens, current_accumulation_step=None, train_mode=None):
        del output_tokens, train_mode
        t = self._sample_t(x0.shape[0], current_accumulation_step)
        dt = self._compute_dt()
        t = torch.clip(t + dt, 0, 1)

        gamma_t = self.gamma_min + t * (self.gamma_max - self.gamma_min)
        gamma_s = self.gamma_min + (t - dt) * (self.gamma_max - self.gamma_min)

        usdm_alpha_t = self._gamma_to_alphat(gamma_t)
        usdm_alpha_t = usdm_alpha_t.unsqueeze(-1)
        assert usdm_alpha_t.ndim == 2
        usdm_alpha_s = self._gamma_to_alphat(gamma_s)
        usdm_alpha_s = usdm_alpha_s.unsqueeze(-1)
        assert usdm_alpha_s.ndim == 2

        xt, xs = self._sample_trajectory(x0, gamma_t, gamma_s)
        xt_discrete = xt.argmax(-1)
        xs_discrete = xs.argmax(-1)
        log_x_theta_student = self.forward(
            xt_discrete, sigma=self._sigma_from_alphat(usdm_alpha_t)
        )
        log_x_theta_teacher = self._teacher_logits(
            xs_discrete, sigma=self._sigma_from_alphat(usdm_alpha_s)
        )
        if self.config.training.loss_precision == "float64":
            log_x_theta_student = log_x_theta_student.to(torch.float64)
            log_x_theta_teacher = log_x_theta_teacher.to(torch.float64)
        if self.loss_type == "kl-fwd":
            return (
                log_x_theta_teacher.exp() * (log_x_theta_teacher - log_x_theta_student)
            ).sum(-1)
        elif self.loss_type == "kl-bwd":
            return (
                log_x_theta_student.exp() * (log_x_theta_student - log_x_theta_teacher)
            ).sum(-1)

    def training_step(self, batch, batch_idx):
        self.log(
            name="dt",
            value=self._compute_dt(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return super().training_step(batch, batch_idx)
