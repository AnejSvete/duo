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
        number of levels masked (discrete timesteps). Otherwise, `t` is sampled randomly.
        """
        del output_tokens

        if not ground_truth_masking:
            # --- Standard Path: Sample t first, then create xt ---
            t = self._sample_t(x0.shape[0], current_accumulation_step)
            if self.T > 0:
                t = (t * self.T).to(torch.int) / self.T + (1 / self.T)

            dalpha_t, alpha_t = self.noise(t)
            xt, _, active_mask = self.q_xt(
                x0, alpha_t.unsqueeze(-1), do_not_mask, ground_truth_masking=False
            )
        else:
            # --- Ground Truth Path: Create xt first, then derive t from discrete levels ---
            # 1. Get the noisy sample, the number of levels, and the active mask.
            #    alpha_t is not used by this q_xt path, so we pass None.
            xt, num_levels, active_mask = self.q_xt(
                x0, alpha_t=None, do_not_mask=do_not_mask, ground_truth_masking=True
            )

            # 2. Calculate how many levels are currently masked (from the xt output).
            #    Count the masked tokens in each sequence to infer the level.
            mask_counts = (xt == self.mask_index).sum(dim=1).float()
            num_maskable_tokens = (~do_not_mask).sum(dim=1).float()
            num_maskable_tokens = torch.clamp(num_maskable_tokens, min=1.0)  # Avoid division by zero

            # Estimate which timestep (level) this corresponds to
            # t should be uniformly distributed over the discrete levels: t = level / num_levels
            mask_ratio = (mask_counts / num_maskable_tokens).clamp(0.0, 1.0)

            # Map mask_ratio to discrete timesteps based on levels
            # For each sequence, t = (levels_masked / total_levels)
            # Since we sample uniformly from 1 to num_levels, we approximate:
            t = mask_ratio.clamp(min=1.0 / self.T if self.T > 0 else 1e-6)

            # 3. Compute the noise schedule variables from the derived t.
            dalpha_t, alpha_t = self.noise(t)

            # 4. Adjust dalpha_t for discrete timesteps
            # Since we have discrete levels, the weight should reflect the discrete nature
            # dalpha_t represents the "density" at this timestep
            # For uniform sampling over K levels: weight = 1/K for each level
            # We scale dalpha_t by the number of levels
            level_weights = 1.0 / num_levels.float().clamp(min=1.0)
            dalpha_t = dalpha_t * level_weights

        # --- Common Logic for both paths ---
        alpha_t_unsqueezed = alpha_t.unsqueeze(-1)
        dalpha_t_unsqueezed = dalpha_t.unsqueeze(-1)
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

            # Log ground truth masking specific metrics
            if ground_truth_masking:
                self.log("diffusion/ground_truth_masking", 1.0, on_step=True, on_epoch=False, sync_dist=True)
                # Log level-based statistics if available
                # Note: num_levels is computed in q_xt but not returned in standard path
                # We can compute it here for logging
                pipe_token_id = self.tokenizer.convert_tokens_to_ids("|")
                num_pipes_per_seq = (x0 == pipe_token_id).sum(dim=1).float()
                self.log("diffusion/num_levels_mean", num_pipes_per_seq.mean(), on_step=True, on_epoch=False, sync_dist=True)
                self.log("diffusion/num_levels_std", num_pipes_per_seq.std(), on_step=True, on_epoch=False, sync_dist=True)

                # Log active mask statistics
                if active_mask is not None:
                    active_ratio = active_mask.float().mean()
                    self.log("diffusion/active_ratio", active_ratio, on_step=True, on_epoch=False, sync_dist=True)
                    active_ratio_per_seq = active_mask.float().mean(dim=1)
                    self.log("diffusion/active_ratio_per_seq_mean", active_ratio_per_seq.mean(), on_step=True, on_epoch=False, sync_dist=True)
                    self.log("diffusion/active_ratio_per_seq_std", active_ratio_per_seq.std(), on_step=True, on_epoch=False, sync_dist=True)

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
            dalpha_t=dalpha_t_unsqueezed,
            low_var=train_mode and self.loss_type == "low_var",
        )

        # If using ground_truth_masking, only compute loss on the active level
        if ground_truth_masking and active_mask is not None:
            # Zero out loss for tokens not in the active level
            # This ensures only the "next level" tokens contribute to the loss
            nll_result = nll_result * active_mask.float()

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

        If ground_truth_masking is True, it masks levels (segments between '|' delimiters)
        from right to left, based on a uniformly sampled discrete timestep. This is designed
        for hierarchical structured tasks like BFVP and arithmetic where computation proceeds
        in levels.

        For a sequence like: "# A | B | C | D"
        - There are 3 pipes, representing 3 levels (the regions after each pipe)
        - Masking 1 level: "# A | B | C | [MASK D]"
        - Masking 2 levels: "# A | B | [MASK C] [MASK D]"
        - Masking 3 levels: "# A | [MASK B] [MASK C] [MASK D]"

        The timestep t is uniformly sampled from {1, 2, ..., num_levels}, where num_levels
        is the number of pipes. This gives at most ceil(log(N)) discrete timesteps for a
        sequence of length N.

        Returns:
            A tuple of (xt, num_levels, active_mask), where:
            - num_levels is a tensor of the total number of levels in each sequence for ground_truth_masking and None otherwise.
            - active_mask is a boolean tensor indicating which tokens are in the "active" level (the most recently masked level)
              for ground_truth_masking, and None otherwise. This is used to compute loss only on the active level.
        """
        if not ground_truth_masking:
            # Standard probabilistic masking based on the noise schedule.
            potential_mask = torch.rand(*x.shape, device=x.device) < 1 - alpha_t
            final_mask = potential_mask & ~do_not_mask
            xt = torch.where(final_mask, self.mask_index, x)

            # Return None for num_levels and active_mask in the standard case.
            return xt, None, None
        else:
            # Ground truth masking: mask levels from right to left based on timestep.
            xt = x.clone()
            batch_size, seq_len = x.shape
            num_levels = torch.zeros(
                batch_size, device=x.device, dtype=torch.long
            )
            # Track which tokens belong to the "active" level (the one we just masked)
            active_mask = torch.zeros(
                batch_size, seq_len, device=x.device, dtype=torch.bool
            )

            pipe_token_id = self.tokenizer.convert_tokens_to_ids("|")

            if pipe_token_id == self.tokenizer.unk_token_id:
                raise ValueError(
                    "The '|' character is not in the tokenizer's vocabulary."
                )

            for i in range(batch_size):
                # Find all pipe positions in this sequence (that are not in do_not_mask region)
                pipe_indices = (x[i] == pipe_token_id).nonzero(as_tuple=True)[0]
                valid_pipe_indices = pipe_indices[~do_not_mask[i][pipe_indices]]

                if len(valid_pipe_indices) == 0:
                    continue

                # Number of levels = number of segments between pipes + 1 for the final segment
                # For example: "# A | B | C" has 3 pipes total, but the completion has 3 levels (A, B, C)
                total_levels = len(valid_pipe_indices)
                if total_levels == 0:
                    continue

                num_levels[i] = total_levels

                # Sample which timestep (how many levels to mask from the right)
                # timestep ranges from 1 to total_levels (mask at least 1 level)
                # Uniformly sample: each level has equal probability
                levels_to_mask = torch.randint(1, total_levels + 1, (1,)).item()

                # Determine the active level range (the first level being masked)
                # For example, if levels_to_mask = 2 for "# A | B | C | D", we mask C and D
                # The "active" level is C (the first one being masked at this timestep)
                if levels_to_mask == total_levels:
                    # Masking all levels - the active level is the first level after '#'
                    hash_token_id = self.tokenizer.convert_tokens_to_ids("#")
                    hash_indices = (x[i] == hash_token_id).nonzero(as_tuple=True)[0]
                    if len(hash_indices) > 0:
                        active_start = hash_indices[-1].item() + 1  # Start after the last '#'
                    else:
                        active_start = 0

                    # The active level ends at the first pipe
                    if len(valid_pipe_indices) > 0:
                        active_end = valid_pipe_indices[0].item()
                    else:
                        active_end = seq_len
                elif levels_to_mask == 1:
                    # Masking only the last level - active level is after the last pipe
                    active_start = valid_pipe_indices[-1].item()
                    active_end = seq_len
                else:
                    # Masking k levels from the right
                    # The active level is between pipe at index (total_levels - levels_to_mask)
                    # and pipe at index (total_levels - levels_to_mask + 1)
                    pipe_idx = total_levels - levels_to_mask
                    active_start = valid_pipe_indices[pipe_idx].item()
                    if pipe_idx + 1 < len(valid_pipe_indices):
                        active_end = valid_pipe_indices[pipe_idx + 1].item()
                    else:
                        active_end = seq_len

                # Mark the active level in the mask
                for j in range(active_start, active_end):
                    if not do_not_mask[i, j]:
                        active_mask[i, j] = True

                # Mask from right to left, starting from the (total_levels - levels_to_mask)th pipe
                # Index into valid_pipe_indices from the right
                if levels_to_mask == total_levels:
                    # Mask everything after '#'
                    hash_token_id = self.tokenizer.convert_tokens_to_ids("#")
                    hash_indices = (x[i] == hash_token_id).nonzero(as_tuple=True)[0]
                    if len(hash_indices) > 0:
                        start_pos = hash_indices[-1].item() + 1  # Start after the last '#'
                    else:
                        start_pos = 0

                    # Mask everything from start_pos to end
                    for j in range(start_pos, seq_len):
                        if not do_not_mask[i, j]:
                            xt[i, j] = self.mask_index
                else:
                    # Mask from the pipe that separates the levels
                    # If we want to mask k levels from the right, we start from pipe at index -(k)
                    pipe_idx = total_levels - levels_to_mask
                    if pipe_idx >= 0 and pipe_idx < len(valid_pipe_indices):
                        # Start masking from this pipe position (inclusive of the pipe)
                        start_mask_pos = valid_pipe_indices[pipe_idx].item()

                        # Mask from start_mask_pos to end
                        for j in range(start_mask_pos, seq_len):
                            if not do_not_mask[i, j]:
                                xt[i, j] = self.mask_index

            # Return the modified sequence, the number of levels, and the active mask.
            return xt, num_levels, active_mask

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

    def q_xt(self, x, alpha_t, do_not_mask, ground_truth_masking=False):
        """Computes the noisy sample xt, protecting specified tokens."""
        # UniformState doesn't support ground_truth_masking
        if ground_truth_masking:
            raise NotImplementedError("UniformState does not support ground_truth_masking")

        # Decide which tokens to potentially corrupt based on the noise schedule
        potential_corruption = torch.rand(*x.shape, device=x.device) < 1 - alpha_t

        # Only corrupt tokens where potential_corruption is True AND do_not_mask is False
        final_corruption = potential_corruption & ~do_not_mask

        uniform_tensor = torch.randint(0, self.vocab_size, x.shape, device=x.device)

        xt = torch.where(final_corruption, uniform_tensor, x)
        return xt, None, None

    def prior_sample(self, *batch_dims):
        return torch.randint(
            0, self.vocab_size, batch_dims, dtype=torch.int64, device=self.device
        )
