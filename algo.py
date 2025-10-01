import numpy as np
import torch

import trainer_base


class AR(trainer_base.TrainerBase):
    def generate_conditioned(self, prompts, targets, mode="random", top_k=1):
        """
        Generate completions conditioned on prompts using efficient, vectorized
        autoregressive decoding. This version uses a deterministic greedy strategy.

        prompts: (batch, seq) tensor (padded with pad_token_id)
        targets: (batch, seq) tensor (padded with pad_token_id)
        Returns: (batch, seq) tensor containing the prompts and generated completions.
        """
        _, seq_len = prompts.shape

        # Start with the prompts. We will fill this tensor one token at a time.
        x = prompts.clone()

        # Pre-calculate the length of each prompt to know where generation starts.
        # Find the index of the first occurrence of the mask token in each sequence.
        # If no mask token is present, set to seq_len (i.e., generate nothing).
        mask_positions = prompts == self.mask_index
        prompt_lens = torch.where(
            mask_positions.any(dim=1),
            mask_positions.float().argmax(dim=1),
            torch.full((prompts.shape[0],), seq_len, device=prompts.device),
        )

        # Autoregressively generate tokens for each position in the sequence.
        for t in range(min(prompt_lens), seq_len):
            # A mask to determine which samples need a token generated at this step.
            mask_to_generate = t >= prompt_lens

            # If no samples need a token at this position, skip to the next step.
            if not mask_to_generate.any():
                continue

            # Get model predictions based on all preceding tokens.
            with torch.no_grad():
                # The input is the sequence so far: x[:, :t].
                logits = self.backbone(x[:, :t], None)  # Shape: (B, t, V)

                # We only need the logits for the very last position in the sequence.
                last_token_logits = logits[:, -1, :]  # Shape: (B, V)

                probs = last_token_logits.softmax(dim=-1)

            # For AR generation, deterministically select the most likely token (greedy decoding).
            next_tok = probs.argmax(dim=-1)

            # Place the newly generated token at position 't' for the active samples.
            x[mask_to_generate, t] = next_tok[mask_to_generate]

        return x

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

    def _process_model_input(self, x0, valid_tokens):
        input_tokens = x0[:, :-1]
        output_tokens = x0[:, 1:]
        valid_tokens = valid_tokens[:, 1:]
        return input_tokens, output_tokens, valid_tokens

    def nll(
        self,
        input_tokens,
        output_tokens,
        do_not_mask,
        current_accumulation_step,
        train_mode,
        ground_truth_masking,
    ):
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


class LT(trainer_base.TrainerBase):
    def generate_conditioned(self, prompts, targets=None, **kwargs):
        """
        Generate symbols for all masked positions at once using a deterministic
        greedy decoding strategy. This is a non-autoregressive, single-step process.

        Args:
            prompts (torch.Tensor): A (batch, seq) tensor with masked positions
                                    indicated by `self.mask_index`.
            targets (torch.Tensor, optional): This parameter is ignored.
            **kwargs: Any additional keyword arguments (like 'mode' or 'top_k') are ignored.

        Returns:
            torch.Tensor: A (batch, seq) tensor with the masked positions filled in.
        """
        # Clone the input to avoid modifying the original tensor.
        filled_sequence = prompts.clone()
        mask_to_generate = prompts == self.mask_index

        # Return early if there are no masks to fill.
        if not mask_to_generate.any():
            return filled_sequence

        # Perform a single forward pass to get predictions for all token positions.
        with torch.no_grad():
            logits = self.backbone(filled_sequence, None)
            logits[:, :, self.mask_index] = self.neg_infinity
            logits[:, :, self.tokenizer.pad_token_id] = self.neg_infinity

        # Deterministically select the most likely token for every position.
        predicted_tokens = torch.argmax(logits, dim=-1)

        # Fill the masked positions with the corresponding predictions.
        filled_sequence[mask_to_generate] = predicted_tokens[mask_to_generate]

        return filled_sequence

    def __init__(self, config, tokenizer):
        vocab_size = tokenizer.vocab_size
        if not hasattr(tokenizer, "mask_token") or tokenizer.mask_token is None:
            self.mask_index = vocab_size
            vocab_size += 1
        else:
            self.mask_index = tokenizer.mask_token_id
        # The base class __init__ needs to be called.
        # Using placeholder values for config and backbone for this example.
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()
        self._validate_configuration()

    def _validate_configuration(self):
        super()._validate_configuration()
        # Mocking config attributes for demonstration
        if not hasattr(self.config, "algo"):
            self.config.algo = type("obj", (object,), {"time_conditioning": False})()
        assert not self.config.algo.time_conditioning

    def _process_model_input(self, x0, valid_tokens):
        """
        Prepares model input by masking the label region in a vectorized way.

        This function identifies a label region in each sequence and replaces all
        *non-padding* tokens within that region with a mask token.
        The label region is defined as the tokens between the '#' delimiter and
        the start of the final contiguous block of padding tokens.

        This approach robustly handles formats like:
        1. ... X X # Y Y Y [PAD] [PAD]       -> Masks 'Y Y Y'
        2. ... X X # [PAD] Y [PAD] [PAD]   -> Masks only 'Y', leaves [PAD]
        """
        # Create a copy to modify for the input, keeping x0 as the target.
        input_tokens = x0.clone()

        # Get special token IDs.
        try:
            hash_token_id = self.tokenizer.convert_tokens_to_ids("#")
        except KeyError:
            # If '#' is not in the vocabulary, no masking can be done.
            return input_tokens, x0, valid_tokens

        pad_token_id = self.tokenizer.pad_token_id
        seq_len = x0.shape[1]

        # --- Vectorized Masking Logic ---

        # 1. Identify sequences that contain the '#' delimiter.
        is_hash = x0 == hash_token_id
        has_hash = is_hash.any(dim=1)

        if not has_hash.any():
            return input_tokens, x0, valid_tokens

        # 2. Find the start and end boundaries for the label region.
        hash_indices = torch.argmax(is_hash.int(), dim=1)
        is_not_pad = x0 != pad_token_id
        last_non_pad_indices = (
            seq_len - 1 - torch.argmax(torch.flip(is_not_pad, dims=[1]).int(), dim=1)
        )

        # 3. Build the boolean mask for tokens to be replaced.
        col_indices = torch.arange(seq_len, device=x0.device).expand_as(x0)

        # A token is masked if it's:
        # (A) in a sequence that has a '#'
        # (B) located *after* the '#'
        # (C) located *at or before* the last non-padding token
        # (D) AND is not a padding token itself
        final_mask = (
            has_hash.unsqueeze(1)
            & (col_indices > hash_indices.unsqueeze(1))
            & (col_indices <= last_non_pad_indices.unsqueeze(1))
            & (x0 != pad_token_id)  # <-- This is the new condition
        )

        # 4. Apply the mask to replace the target tokens with the mask index.
        input_tokens[final_mask] = self.mask_index

        return input_tokens, x0, valid_tokens

    def nll(
        self,
        input_tokens,
        output_tokens,
        do_not_mask,
        current_accumulation_step,
        train_mode,
        ground_truth_masking,
    ):
        del current_accumulation_step
        output = self.backbone(input_tokens, None)
        output = output.log_softmax(-1)

        nll_per_token = -output.gather(-1, output_tokens[:, :, None])[:, :, 0]

        mlm_mask = input_tokens == self.mask_index

        masked_nll = nll_per_token * mlm_mask

        return masked_nll

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

    def generate_conditioned(self, prompts, targets, mode="random", top_k=1):
        """
        Generate completions conditioned on prompts, using the specified unmasking mode.
        prompts: (batch, seq) tensor (padded)
        targets: (batch, seq) tensor (padded)
        Returns: (batch, seq) tensor (same shape as prompts)
        """
        batch_size, seq_len = prompts.shape

        x = prompts.clone()

        # Tracks which sequences in the batch are complete.
        finished = torch.zeros(batch_size, dtype=torch.bool, device=prompts.device)

        prompt_lens = (
            (prompts != self.tokenizer.pad_token_id) & (prompts != self.mask_index)
        ).sum(dim=1)

        # The main generation loop continues as long as there are masks to fill.
        for tstep in range(seq_len):
            # Identify mask positions for the entire batch.
            mask_pos = x == self.mask_index

            # If no masks are left anywhere in the batch, we can stop.
            if not mask_pos.any():
                break

            # Get model predictions for the entire batch.
            with torch.no_grad():
                # Use sigma=0 for the standard denoising/generation step.
                sigma = torch.zeros(batch_size, device=prompts.device)
                logits = self.backbone(x, sigma)
                # Do not predict mask or padding tokens
                logits[:, :, self.mask_index] = -torch.inf
                logits[:, :, self.tokenizer.pad_token_id] = -torch.inf
                probs = logits.softmax(dim=-1)

            # Create a mask for rows that are not yet finished.
            unfinished_mask = ~finished

            if mode == "random":
                # For each unfinished sequence, pick top_k random masked positions to fill.

                # 1. Create random weights for all positions.
                rand_weights = torch.rand(x.shape, device=prompts.device)

                # 2. Ignore non-masked positions by setting their weights to a negative value.
                rand_weights[~mask_pos] = -1.0

                # 3. Determine the actual number of positions to fill for each sequence.
                # This is the minimum of top_k and the number of available masks.
                num_masks_per_item = mask_pos.sum(dim=1)
                # Ensure k is not larger than the sequence length to avoid errors with topk.
                k = min(top_k, x.shape[1])
                actual_k = torch.min(
                    torch.tensor(k, device=prompts.device), num_masks_per_item
                )

                # 4. Find the indices of the top_k largest random weights for each row.
                # These are our randomly chosen positions. We run topk with a fixed k
                # and will only use the valid number of positions for each item later.
                _, topk_pos = torch.topk(rand_weights, k=k, dim=1)

                # 5. Gather the probability distributions at these k chosen positions.
                # The result `gathered_probs` will have shape (batch_size, k, vocab_size).
                gathered_probs = torch.gather(
                    probs, 1, topk_pos.unsqueeze(-1).expand(-1, -1, probs.shape[-1])
                )

                # 6. Sample one token for each of the k chosen positions.
                # We reshape for multinomial and then reshape back to (batch_size, k).
                reshaped_probs = gathered_probs.view(-1, probs.shape[-1])
                # Add a small epsilon to prevent errors if probabilities sum to zero.
                sampled_indices = torch.multinomial(
                    reshaped_probs + 1e-9, num_samples=1
                )
                chosen_tokens = sampled_indices.view(x.shape[0], k)

                # 7. Place the new tokens into `x` at the chosen positions.
                # We iterate because the number of tokens to update (actual_k) can vary
                # for each sequence in the batch.
                rows_to_update = unfinished_mask.nonzero(as_tuple=True)[0]
                for i in rows_to_update:
                    # Get the number of valid positions to update for this specific sequence.
                    num_valid = actual_k[i].item()
                    if num_valid > 0:
                        # Select the specific positions and tokens for this sequence.
                        positions = topk_pos[i, :num_valid]
                        values = chosen_tokens[i, :num_valid]
                        # Update the main tensor `x` at the chosen positions.
                        x[i, positions] = values

            elif mode == "top_k":
                # For each unfinished sequence, fill the top_k most confident masked positions.

                confidences, best_tokens = probs.max(dim=-1)
                confidences[~mask_pos] = -1.0

                num_masks_per_item = mask_pos.sum(dim=1)
                k = min(top_k, confidences.shape[1])
                actual_k = torch.min(
                    torch.tensor(k, device=prompts.device), num_masks_per_item
                )

                if actual_k.max() > 0:
                    _, topk_pos = torch.topk(confidences, k=k, dim=1)
                    tokens_to_insert = torch.gather(best_tokens, 1, topk_pos)

                    # Only update valid top-k positions for unfinished sequences
                    for i in range(batch_size):
                        if not unfinished_mask[i]:
                            continue
                        num_valid = actual_k[i].item()
                        if num_valid > 0:
                            positions = topk_pos[i, :num_valid]
                            values = tokens_to_insert[i, :num_valid]
                            x[i, positions] = values

            elif mode == "one_level":
                for i in range(batch_size):

                    if finished[i]:
                        continue

                    start = prompt_lens[i].item()

                    bars = (
                        targets[i][start:] == self.tokenizer.convert_tokens_to_ids("|")
                    ).nonzero(as_tuple=True)

                    end = (
                        start + bars[0][0].item() + 1 if len(bars[0]) > 0 else start + 1
                    )

                    # Only perform the update if the slice is valid.
                    if start < end:
                        x[i, start:end] = probs[i, start:end].argmax(dim=-1)

                    prompt_lens[i] = end

            elif mode == "one_at_a_time":
                # For each unfinished sequence, fill the single, left-most masked position.

                # Get the indices of sequences that still have masks.
                rows_to_update = unfinished_mask.nonzero(as_tuple=True)[0]

                if rows_to_update.numel() > 0:
                    # Find the position of the first mask for each of these active sequences.
                    first_mask_indices = (
                        (x[rows_to_update] == self.mask_index).float().argmax(dim=1)
                    )

                    # Get the probability distributions at these specific positions.
                    probs_to_use = probs[rows_to_update, first_mask_indices, :]

                    # Select the most likely token for each position (greedy decoding).
                    next_tokens = probs_to_use.argmax(dim=-1)

                    # Place the newly generated tokens into the correct positions in `x`.
                    x[rows_to_update, first_mask_indices] = next_tokens

            elif mode == "all_at_once":

                for i in range(batch_size):

                    if finished[i]:
                        continue

                    start = prompt_lens[i].item()

                    bars = (
                        targets[i][start:] == self.tokenizer.convert_tokens_to_ids("|")
                    ).nonzero(as_tuple=True)

                    end = (
                        start + bars[0][-1].item() + 2
                        if len(bars[0]) > 0
                        else start + 2
                    )

                    # Only perform the update if the slice is valid.
                    if start < end:
                        x[i, start:end] = probs[i, start:end].argmax(dim=-1)

                    prompt_lens[i] = end

            else:
                raise ValueError(f"Unknown generation mode: {mode}")

            # Update the finished status for any sequences that no longer have masks.
            finished |= (x != self.mask_index).all(dim=1)
            if finished.all():
                break

        return x


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
