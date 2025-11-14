import numpy as np
import torch
import torch.nn.functional as F

from trainer_base import TrainerBase


def sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


class Diffusion(TrainerBase):
    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.config.sampling.noise_removal in {"none", "ancestral", "greedy"}
        assert self.loss_type in {"elbo", "low_var"}
        if self.config.sampling.noise_removal == "greedy":
            assert self.sampler != "analytic"
            assert self.parameterization in {"mean", "subs"}

    def _process_model_input(self, x0, valid_tokens):
        return x0, None, valid_tokens

    def _process_sigma(self, sigma):
        assert sigma.ndim == 2
        sigma = sigma.mean(-1).squeeze()
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, n, accum_step):
        """
        Samples timesteps `t` for a batch of size `n`.

        If training with antithetic sampling and gradient accumulation, it generates
        stratified samples for the entire global batch and returns the appropriate
        chunk for the current accumulation step, correctly sized to `n`.
        """
        # For validation, or if not using antithetic sampling, use simple random sampling.
        if accum_step is None or not self.antithetic_sampling:
            _eps_t = torch.rand(n, device=self.device)
            t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
            return t

        # Handle the training case with antithetic sampling and gradient accumulation.
        else:
            global_batch_size = self.config.loader.global_batch_size
            num_accum_steps = self.trainer.accumulate_grad_batches

            _eps_t_global = torch.rand(global_batch_size, device=self.device)
            offset_global = (
                torch.arange(global_batch_size, device=self.device) / global_batch_size
            )
            t_global = (_eps_t_global / global_batch_size + offset_global) % 1.0

            chunks = t_global.chunk(num_accum_steps)

            # Check for valid accumulation step to prevent index errors.
            if accum_step >= len(chunks):
                # Fallback to the first chunk if accum_step is somehow out of range.
                accum_step = 0

            current_chunk = chunks[accum_step]

            sized_chunk = current_chunk[:n]

            t = (1 - self.sampling_eps) * sized_chunk + self.sampling_eps
            return t

    def _sigma_from_alphat(self, alpha_t):
        return -torch.log(alpha_t)

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(1, x0.shape[0], dtype=self.dtype, device=self.device)
        sigma_t0 = self._sigma_from_alphat(self.noise(t0)[1])
        model_output_t0 = self.forward(x0, sigma_t0)
        return -torch.gather(
            input=model_output_t0, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

    def nll_per_token(self, model_output, xt, x0, alpha_t, dalpha_t, low_var):
        raise NotImplementedError

    def nll(
        self,
        x0,
        output_tokens,
        do_not_mask,
        current_accumulation_step=None,
        train_mode=False,
        ground_truth_masking=False,
    ):
        """
        Calculates the Negative Log-Likelihood loss for a batch.

        If ground_truth_masking is True, the noise level `t` is derived from the
        number of masked tokens. Otherwise, `t` is sampled randomly.
        """
        del output_tokens

        if not ground_truth_masking:
            # --- Standard Path: Sample t first, then create xt ---
            t = self._sample_t(x0.shape[0], current_accumulation_step)
            if self.T > 0:
                t = (t * self.T).to(torch.int) / self.T + (1 / self.T)

            dalpha_t, alpha_t = self.noise(t)
            xt, _ = self.q_xt(
                x0, alpha_t.unsqueeze(-1), do_not_mask, ground_truth_masking=False
            )
        else:
            # --- Ground Truth Path: Create xt first, then derive t ---
            # 1. Get the noisy sample and the number of masked tokens.
            #    alpha_t is not used by this q_xt path, so we pass None.
            xt, masked_counts = self.q_xt(
                x0, alpha_t=None, do_not_mask=do_not_mask, ground_truth_masking=True
            )

            # 2. Calculate the actual mask ratio for each sequence.
            num_maskable_tokens = (~do_not_mask).sum(dim=1)
            num_maskable_tokens[num_maskable_tokens == 0] = (
                1.0  # Avoid division by zero.
            )
            mask_ratio = (masked_counts / num_maskable_tokens).clamp(0.0, 1.0)

            # 3. Derive t from the mask_ratio (linear schedule: t = mask_ratio).
            #    Ensure t is in the valid range, e.g., [1/T, 1] if required by the schedule.
            t = mask_ratio.clamp(min=1.0 / self.T if self.T > 0 else 1e-6)

            # 4. Compute the noise schedule variables from the derived t.
            dalpha_t, alpha_t = self.noise(t)

        # --- Common Logic for both paths ---
        alpha_t_unsqueezed = alpha_t.unsqueeze(-1)
        sigma = self._sigma_from_alphat(alpha_t_unsqueezed)

        # Log diffusion-specific metrics during training
        if train_mode and self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            # Log alpha_t statistics
            self.log("diffusion/alpha_t_mean", alpha_t.mean(), on_step=True, on_epoch=False, sync_dist=True)
            self.log("diffusion/alpha_t_std", alpha_t.std(), on_step=True, on_epoch=False, sync_dist=True)
            self.log("diffusion/alpha_t_min", alpha_t.min(), on_step=True, on_epoch=False, sync_dist=True)
            self.log("diffusion/alpha_t_max", alpha_t.max(), on_step=True, on_epoch=False, sync_dist=True)

            # Log sigma statistics
            self.log("diffusion/sigma_mean", sigma.mean(), on_step=True, on_epoch=False, sync_dist=True)
            self.log("diffusion/sigma_std", sigma.std(), on_step=True, on_epoch=False, sync_dist=True)

            # Log timestep statistics (t values)
            self.log("diffusion/t_mean", t.mean(), on_step=True, on_epoch=False, sync_dist=True)
            self.log("diffusion/t_std", t.std(), on_step=True, on_epoch=False, sync_dist=True)

            # Log masking ratio (percentage of tokens masked)
            if hasattr(self, 'mask_index'):
                mask_ratio = (xt == self.mask_index).float().mean()
                self.log("diffusion/mask_ratio", mask_ratio, on_step=True, on_epoch=False, sync_dist=True)

                # Log masking ratio per sequence (useful for ground_truth_masking)
                mask_ratio_per_seq = (xt == self.mask_index).float().mean(dim=1)
                self.log("diffusion/mask_ratio_per_seq_mean", mask_ratio_per_seq.mean(), on_step=True, on_epoch=False, sync_dist=True)
                self.log("diffusion/mask_ratio_per_seq_std", mask_ratio_per_seq.std(), on_step=True, on_epoch=False, sync_dist=True)

        log_x_theta = self.forward(xt, sigma=sigma)

        # Check for NaN/Inf in model output
        if train_mode and (torch.isnan(log_x_theta).any() or torch.isinf(log_x_theta).any()):
            print(f"\n{'='*80}")
            print(f"WARNING: Invalid model output detected at global_step={self.global_step}")
            print(f"{'='*80}")
            print(f"NaN detected: {torch.isnan(log_x_theta).any()}")
            print(f"Inf detected: {torch.isinf(log_x_theta).any()}")
            print(f"alpha_t stats - min: {alpha_t.min():.6f}, max: {alpha_t.max():.6f}, mean: {alpha_t.mean():.6f}")
            print(f"sigma stats - min: {sigma.min():.6f}, max: {sigma.max():.6f}, mean: {sigma.mean():.6f}")
            print(f"t stats - min: {t.min():.6f}, max: {t.max():.6f}, mean: {t.mean():.6f}")
            print(f"dalpha_t stats - min: {dalpha_t.min():.6f}, max: {dalpha_t.max():.6f}, mean: {dalpha_t.mean():.6f}")
            print(f"{'='*80}\n")
            # Log to W&B
            self.log("debug/has_nan", 1.0, on_step=True, on_epoch=False, sync_dist=True)
            self.log("debug/has_inf", 1.0, on_step=True, on_epoch=False, sync_dist=True)

        nll_result = self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t_unsqueezed,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == "low_var",
        )

        # Check for NaN/Inf in loss
        if train_mode and (torch.isnan(nll_result).any() or torch.isinf(nll_result).any()):
            print(f"\n{'='*80}")
            print(f"WARNING: Invalid loss detected at global_step={self.global_step}")
            print(f"{'='*80}")
            print(f"NaN in loss: {torch.isnan(nll_result).any()}")
            print(f"Inf in loss: {torch.isinf(nll_result).any()}")
            print(f"loss stats - min: {nll_result[~torch.isnan(nll_result) & ~torch.isinf(nll_result)].min():.6f}")
            print(f"loss stats - max: {nll_result[~torch.isnan(nll_result) & ~torch.isinf(nll_result)].max():.6f}")
            print(f"{'='*80}\n")
            self.log("debug/loss_has_nan", 1.0, on_step=True, on_epoch=False, sync_dist=True)
            self.log("debug/loss_has_inf", 1.0, on_step=True, on_epoch=False, sync_dist=True)

        return nll_result

    def _get_score(self, **kwargs):
        del kwargs
        raise NotImplementedError

    def _denoiser_update(self, x, t):
        raise NotImplementedError

    def _analytic_update(self, x, t, dt):
        raise NotImplementedError

    def _ancestral_update(self, x, t, dt, p_x0, noise_removal_step):
        raise NotImplementedError

    @torch.no_grad()
    def generate_samples(self, num_samples, num_steps=None, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps
        x = self.prior_sample(num_samples, self.num_tokens)
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ancestral":
                _, x = self._ancestral_update(x=x, t=t, dt=dt, p_x0=None)
            elif self.sampler == "ancestral_cache":
                p_x0_cache, x_next = self._ancestral_update(
                    x=x, t=t, dt=dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x=x, t=t, dt=dt)

        t0 = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
        if self.config.sampling.noise_removal == "ancestral":
            if self.sampler == "analytic":
                x = self._denoiser_update(x=x, t=t0)
            else:
                _, x = self._ancestral_update(
                    x=x, t=t0, dt=None, p_x0=p_x0_cache, noise_removal_step=True
                )
        elif self.config.sampling.noise_removal == "greedy":
            sigma = self._sigma_from_alphat(self.noise(t0)[1])
            x = self.forward(xt=x, sigma=sigma).argmax(dim=-1)
        return x

    @torch.no_grad
    def _semi_ar_sampler(self, n_samples, stride_length, num_strides, dt=0.001):
        # TODO(subham): Test this method after refactoring.
        ones = torch.ones(n_samples, dtype=self.dtype, device=self.device)

        num_steps = int(1 / dt)
        sampling_steps = 0
        intermediate_tokens = []
        target = None
        for _ in range(num_strides + 1):
            p_x0_cache = None
            x = self.prior_sample(n_samples, self.num_tokens)
            if target is not None:
                x[:, :-stride_length] = target
            for i in range(num_steps + 1):
                p_x0_cache, x_next = self._ancestral_update(
                    x=x, t=(1 - i * dt) * ones, dt=dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    p_x0_cache = None
                    sampling_steps += 1
                x = x_next
            x = self.forward(x, 0 * ones).argmax(dim=-1)
            intermediate_tokens.append(x[:, :stride_length].cpu().numpy())
            target = x[:, stride_length:]

        intermediate_tokens.append(target.cpu().numpy())
        intermediate_text_samples = []
        sequence_lengths = (
            (
                np.concatenate(intermediate_tokens, axis=1)[:, 1:]
                == self.tokenizer.eos_token_id
            ).cumsum(-1)
            == 0
        ).sum(-1)
        for i in range(2, len(intermediate_tokens) + 1):
            intermediate_text_samples.append(
                self.tokenizer.batch_decode(
                    np.concatenate(intermediate_tokens[:i], axis=1)
                )
            )
        return (sampling_steps, intermediate_text_samples, sequence_lengths)

    def restore_model_and_semi_ar_sample(self, stride_length, num_strides, dt=0.001):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        # TODO(subham): Test this method after refactoring.
        self._eval_mode()
        (sampling_steps, samples, sequence_lengths) = self._semi_ar_sampler(
            n_samples=self.config.loader.eval_batch_size,
            stride_length=stride_length,
            num_strides=num_strides,
            dt=dt,
        )
        self._train_mode()
        return sampling_steps, samples, sequence_lengths


class AbsorbingState(Diffusion):
    def __init__(self, config, tokenizer):
        # NOTE: Ideally, we should do
        # vocab_size = len(tokenizer), so that we account
        # for the special tokens added in dataloader.py.
        # But we use tokenizer.vocab_size so as to to be
        # consistent with the prior checkpoints.
        vocab_size = tokenizer.vocab_size
        if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()

    def _validate_configuration(self):
        super()._validate_configuration()
        if self.parameterization in {"score", "mean"}:
            assert self.time_conditioning
        assert not (self.parameterization == "mean" and self.T == 0)
        if self.T > 0:
            assert self.parameterization in {"mean", "subs"}

    def q_xt(self, x, alpha_t, do_not_mask, ground_truth_masking):
        """
        Computes the noisy sample xt, protecting specified tokens.

        If ground_truth_masking is True, it masks a specific segment defined by '|'
        separators, pads the rest, and returns the count of masked tokens. Otherwise,
        it performs standard probabilistic masking.

        Returns:
            A tuple of (xt, masked_counts), where masked_counts is a tensor of
            counts for ground_truth_masking and None otherwise.
        """
        if not ground_truth_masking:
            # Standard probabilistic masking based on the noise schedule.
            potential_mask = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
            final_mask = potential_mask & ~do_not_mask
            xt = torch.where(final_mask, self.mask_index, x)

            # Return None for masked_counts in the standard case.
            return xt, None
        else:
            # Ground truth masking: mask a segment, pad the rest, and count the masks.
            xt = x.clone()
            batch_size, seq_len = x.shape
            masked_counts = torch.zeros(
                batch_size, device=x.device, dtype=torch.float32
            )

            pipe_token_id = self.tokenizer.convert_tokens_to_ids("|")
            pad_token_id = self.tokenizer.pad_token_id

            if pipe_token_id == self.tokenizer.unk_token_id:
                raise ValueError(
                    "The '|' character is not in the tokenizer's vocabulary."
                )

            for i in range(batch_size):
                pipe_indices = (x[i] == pipe_token_id).nonzero(as_tuple=True)[0]
                valid_start_pipes = pipe_indices[~do_not_mask[i][pipe_indices]]

                if len(valid_start_pipes) == 0:
                    continue

                # 1. Pick a random valid separator to start from.
                start_pipe_pos = valid_start_pipes[
                    torch.randint(0, len(valid_start_pipes), (1,))
                ].item()
                start_pos = start_pipe_pos + 1

                # 2. Find the end of the segment (the next pipe).
                end_mask_pos = seq_len
                next_pipes = pipe_indices[pipe_indices > start_pipe_pos]
                if len(next_pipes) > 0:
                    end_mask_pos = next_pipes[0].item()

                # Stop if there's nothing to mask (e.g., two pipes are adjacent).
                if start_pos > end_mask_pos:
                    continue

                # 3. Mask tokens WITHIN the segment, INCLUDING the end pipe.
                for j in range(start_pos, min(end_mask_pos + 1, seq_len)):
                    if not do_not_mask[i, j]:
                        xt[i, j] = self.mask_index
                        masked_counts[i] += 1  # Increment the count.

                # 4. Pad everything AFTER the now-masked end pipe.
                start_pad_pos = end_mask_pos + 1
                if start_pad_pos < seq_len:
                    xt[i, start_pad_pos:] = pad_token_id

            # Return the modified sequence and the counts.
            return xt, masked_counts

    def prior_sample(self, *batch_dims):
        return self.mask_index * torch.ones(
            *batch_dims, dtype=torch.int64, device=self.device
        )

    def _ancestral_update(self, x, t, dt, p_x0=None, noise_removal_step=False):
        _, alpha_t = self.noise(t)
        if noise_removal_step:
            alpha_s = torch.ones_like(alpha_t)
        else:
            _, alpha_s = self.noise(t - dt)
        assert alpha_t.ndim == 2
        if p_x0 is None:
            p_x0 = self.forward(x, self._sigma_from_alphat(alpha_t)).exp()

        q_xs = p_x0 * (alpha_s - alpha_t)[:, :, None]
        q_xs[:, :, self.mask_index] = 1 - alpha_s
        _x = sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, dt):
        sigma_t = self._sigma_from_alphat(self.noise(t)[1])
        sigma_s = self._sigma_from_alphat(self.noise(t - dt)[1])
        dsigma = sigma_t - sigma_s
        score = self._get_score(x, sigma_t)
        if self.config.sampling.use_float64:
            score = score.to(torch.float64)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma = self._sigma_from_alphat(self.noise(t)[1])
        score = self._get_score(x, sigma)
        if self.config.sampling.use_float64:
            score = score.to(torch.float64)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index, 1 - torch.exp(-sigma).squeeze(-1), 0)[
            ..., None
        ]
        return edge


class UniformState(Diffusion):
    def _validate_configuration(self):
        super()._validate_configuration()
        assert self.time_conditioning
        assert self.parameterization == "mean"
        assert self.T == 0

    def q_xt(self, x, alpha_t, do_not_mask, mask_mode="random"):
        """Computes the noisy sample xt, protecting specified tokens."""
        # Decide which tokens to potentially corrupt based on the noise schedule
        potential_corruption = torch.rand(*x.shape, device=x.device) < 1 - alpha_t

        # Only corrupt tokens where potential_corruption is True AND do_not_mask is False
        final_corruption = potential_corruption & ~do_not_mask

        uniform_tensor = torch.randint(0, self.vocab_size, x.shape, device=x.device)

        xt = torch.where(final_corruption, uniform_tensor, x)
        return xt

    def prior_sample(self, *batch_dims):
        return torch.randint(
            0, self.vocab_size, batch_dims, dtype=torch.int64, device=self.device
        )
