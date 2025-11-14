import itertools
import json
import math
import os
from dataclasses import dataclass

import hydra.utils
import lightning as L
import torch
import transformers

import metrics
import models
from length_stratified_metrics import PerGenerationModeMetrics

torch.set_printoptions(
    threshold=float("inf"),  # Print all elements (no truncation)
    linewidth=200,  # Wider lines before wrapping
    precision=4,  # Decimal precision for floats
    sci_mode=False,  # Disable scientific notation
)


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    prior_loss: torch.FloatTensor
    num_tokens: torch.FloatTensor


class LogLinear(torch.nn.Module):
    """
    Log-Linear noise schedule: alpha(t) = 1 - t
    Simple linear decay from 1 to eps.

    Note: This returns dalpha/dt, not just the coefficient.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, t):
        """
        Args:
            t: timestep in [eps, 1]
        Returns:
            dalpha_t: derivative of alpha with respect to t
            alpha_t: signal level at time t
        """
        t = (1 - self.eps) * t + self.eps
        alpha_t = 1 - t
        dalpha_t = -(1 - self.eps) * torch.ones_like(t)
        return dalpha_t, alpha_t


class Cosine(torch.nn.Module):
    """
    Cosine noise schedule from "Improved Denoising Diffusion Probabilistic Models".
    alpha(t) = cos^2(pi * t / 2)

    Provides smoother transitions and better performance on many tasks.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.pi_over_2 = torch.pi / 2

    def forward(self, t):
        """
        Args:
            t: timestep in [eps, 1]
        Returns:
            dalpha_t: derivative of alpha with respect to t
            alpha_t: signal level at time t
        """
        t = (1 - self.eps) * t + self.eps
        # alpha(t) = cos^2(pi * t / 2)
        alpha_t = torch.cos(self.pi_over_2 * t) ** 2
        # dalpha/dt = 2 * cos(pi*t/2) * (-sin(pi*t/2)) * (pi/2)
        #           = -pi * cos(pi*t/2) * sin(pi*t/2)
        #           = -pi/2 * sin(pi*t)
        dalpha_t = -(1 - self.eps) * self.pi_over_2 * torch.sin(torch.pi * t)
        return dalpha_t, alpha_t


class Linear(torch.nn.Module):
    """
    Linear noise schedule (in variance space): beta(t) = beta_min + t * (beta_max - beta_min)
    Then alpha(t) = exp(-integral of beta(t))

    This is the schedule used in the original DDPM paper.
    """

    def __init__(self, beta_min=0.1, beta_max=20.0, eps=1e-3):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.eps = eps

    def forward(self, t):
        """
        Args:
            t: timestep in [eps, 1]
        Returns:
            dalpha_t: derivative of alpha with respect to t
            alpha_t: signal level at time t
        """
        t = (1 - self.eps) * t + self.eps
        # beta(t) = beta_min + t * (beta_max - beta_min)
        beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
        # alpha(t) = exp(-0.5 * (beta_min * t + 0.5 * (beta_max - beta_min) * t^2))
        log_alpha_t = -0.5 * (
            self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        )
        alpha_t = torch.exp(log_alpha_t)
        # dalpha/dt = alpha(t) * d(log_alpha)/dt
        #           = alpha(t) * (-0.5 * (beta_min + (beta_max - beta_min) * t))
        #           = -0.5 * alpha(t) * beta(t)
        dalpha_t = -(1 - self.eps) * 0.5 * alpha_t * beta_t
        return dalpha_t, alpha_t


class Polynomial(torch.nn.Module):
    """
    Polynomial noise schedule: alpha(t) = (1 - t^power)

    - power < 1: More noise early, slow at end (good for coarse-to-fine)
    - power = 1: Linear (same as LogLinear)
    - power > 1: Less noise early, fast at end
    """

    def __init__(self, power=2.0, eps=1e-3):
        super().__init__()
        self.power = power
        self.eps = eps

    def forward(self, t):
        """
        Args:
            t: timestep in [eps, 1]
        Returns:
            dalpha_t: derivative of alpha with respect to t
            alpha_t: signal level at time t
        """
        t = (1 - self.eps) * t + self.eps
        # alpha(t) = 1 - t^power
        alpha_t = 1 - t**self.power
        # dalpha/dt = -power * t^(power-1)
        dalpha_t = -(1 - self.eps) * self.power * t ** (self.power - 1)
        return dalpha_t, alpha_t


class Sigmoid(torch.nn.Module):
    """
    Sigmoid noise schedule: alpha(t) = sigmoid((1-t) * scale - shift)

    Provides smooth transitions with adjustable steepness.
    Can concentrate the diffusion process in a specific time region.
    """

    def __init__(self, scale=6.0, shift=3.0, eps=1e-3):
        super().__init__()
        self.scale = scale
        self.shift = shift
        self.eps = eps

    def forward(self, t):
        """
        Args:
            t: timestep in [eps, 1]
        Returns:
            dalpha_t: derivative of alpha with respect to t
            alpha_t: signal level at time t
        """
        t = (1 - self.eps) * t + self.eps
        # alpha(t) = sigmoid((1-t) * scale - shift)
        x = (1 - t) * self.scale - self.shift
        alpha_t = torch.sigmoid(x)
        # dalpha/dt = sigmoid'(x) * dx/dt
        #           = sigmoid(x) * (1 - sigmoid(x)) * (-scale)
        #           = -scale * alpha(t) * (1 - alpha(t))
        dalpha_t = -(1 - self.eps) * self.scale * alpha_t * (1 - alpha_t)
        return dalpha_t, alpha_t


class SquaredCosine(torch.nn.Module):
    """
    Squared cosine schedule with offset parameter s to prevent alpha from reaching 0 too quickly.
    From "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal 2021).

    alpha(t) = cos^2(pi/2 * (t + s) / (1 + s))

    The offset s controls how quickly alpha decays. Common value: s = 0.008
    """

    def __init__(self, s=0.008, eps=1e-3):
        super().__init__()
        self.s = s
        self.eps = eps
        self.pi_over_2 = torch.pi / 2

    def forward(self, t):
        """
        Args:
            t: timestep in [eps, 1]
        Returns:
            dalpha_t: derivative of alpha with respect to t
            alpha_t: signal level at time t
        """
        t = (1 - self.eps) * t + self.eps
        # alpha_bar(t) = cos^2(pi/2 * (t + s) / (1 + s))
        arg = self.pi_over_2 * (t + self.s) / (1 + self.s)
        alpha_t = torch.cos(arg) ** 2
        # dalpha/dt = 2 * cos(arg) * (-sin(arg)) * d(arg)/dt
        #           = -2 * cos(arg) * sin(arg) * (pi/2) / (1 + s)
        #           = -(pi / (1 + s)) * sin(2 * arg) / 2
        #           = -(pi / (2 * (1 + s))) * sin(pi * (t + s) / (1 + s))
        dalpha_t = (
            -(1 - self.eps) * (torch.pi / (2 * (1 + self.s))) * torch.sin(2 * arg)
        )
        return dalpha_t, alpha_t


class TrainerBase(L.LightningModule):
    def __init__(
        self, config, tokenizer: transformers.PreTrainedTokenizer, vocab_size=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if hasattr(self.config.algo, "loss_type"):
            self.loss_type = config.algo.loss_type
        self.tokenizer = tokenizer
        if vocab_size is None:
            self.vocab_size = len(self.tokenizer)
        else:
            self.vocab_size = vocab_size
        self.sampler = self.config.sampling.predictor
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.parameterization = self.config.algo.parameterization
        if self.config.algo.backbone == "dit":
            self.backbone = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        elif self.config.algo.backbone == "lt":
            if self.config.algo.looping_type == "log":
                loop_depth_function = lambda n: math.ceil(math.log2(n))
            elif self.config.algo.looping_type == "linear":
                loop_depth_function = lambda n: n
            elif self.config.algo.looping_type == "constant":
                loop_depth_function = lambda n: 1
            else:
                raise ValueError(
                    f"Unknown looping type: {self.config.algo.looping_type}"
                )
            self.backbone = models.lt.LT(
                self.config,
                vocab_size=self.vocab_size,
                loop_depth_function=loop_depth_function,
            )

        self.T = self.config.algo.T
        self.num_tokens = self.config.model.length
        self.softplus = torch.nn.Softplus()
        self.p_nucleus = self.config.sampling.p_nucleus

        # Noise Schedule - select based on config
        noise_type = (
            self.config.noise.type if hasattr(self.config, "noise") else "log-linear"
        )
        noise_eps = self.config.noise.eps if hasattr(self.config.noise, "eps") else 1e-3

        if noise_type == "log-linear":
            self.noise = LogLinear(eps=noise_eps)
        elif noise_type == "cosine":
            self.noise = Cosine(eps=noise_eps)
        elif noise_type == "linear":
            beta_min = (
                self.config.noise.beta_min
                if hasattr(self.config.noise, "beta_min")
                else 0.1
            )
            beta_max = (
                self.config.noise.beta_max
                if hasattr(self.config.noise, "beta_max")
                else 20.0
            )
            self.noise = Linear(beta_min=beta_min, beta_max=beta_max, eps=noise_eps)
        elif noise_type == "polynomial":
            power = (
                self.config.noise.power if hasattr(self.config.noise, "power") else 2.0
            )
            self.noise = Polynomial(power=power, eps=noise_eps)
        elif noise_type == "sigmoid":
            scale = (
                self.config.noise.scale if hasattr(self.config.noise, "scale") else 6.0
            )
            shift = (
                self.config.noise.shift if hasattr(self.config.noise, "shift") else 3.0
            )
            self.noise = Sigmoid(scale=scale, shift=shift, eps=noise_eps)
        elif noise_type == "squared-cosine":
            s = self.config.noise.s if hasattr(self.config.noise, "s") else 0.008
            self.noise = SquaredCosine(s=s, eps=noise_eps)
        else:
            raise ValueError(f"Unknown noise schedule type: {noise_type}")

        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=self.config.eval.perplexity_batch_size,
        )

        # Length-stratified metrics for validation and testing (quartiles)
        self.val_length_metrics = PerGenerationModeMetrics(num_bins=4)
        self.test_length_metrics = PerGenerationModeMetrics(num_bins=4)

        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.algo.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None

    def _validate_configuration(self):
        assert self.config.algo.backbone in {"dit", "lt"}
        if self.config.algo.parameterization == "ar":
            assert not self.config.algo.time_conditioning

        if self.parameterization in {"score", "mean"}:
            assert self.time_conditioning
        if self.T > 0:
            assert self.parameterization != "score"

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.metrics.to(*args, **kwargs)
        return self

    def q_xt(self, x, alpha_t, do_not_mask, mask_mode="random"):
        raise NotImplementedError

    def _get_parameters(self):
        return itertools.chain(self.backbone.parameters(), self.noise.parameters())

    def _eval_mode(self):
        self.backbone.eval()
        self.noise.eval()

    def _train_mode(self):
        self.backbone.train()
        self.noise.train()

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)

        # Log gradient norms
        if self.trainer.global_step % self.trainer.log_every_n_steps == 0:
            total_norm = 0.0
            for p in self._get_parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5

            self.log(
                "trainer/grad_norm",
                total_norm,
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

    def _process_sigma(self, sigma):
        raise NotImplementedError

    def _process_model_output(self, model_output, xt, sigma):
        raise NotImplementedError

    def forward(self, xt, sigma):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            model_output = self.backbone(xt, sigma)
        return self._process_model_output(model_output=model_output, xt=xt, sigma=sigma)

    def on_train_epoch_start(self):
        self.metrics.reset()
        assert self.metrics.train_nlls.nll.mean_value == 0
        assert self.metrics.train_nlls.nll.weight == 0

    def training_step(self, batch, batch_idx):
        current_accumulation_step = batch_idx % self.trainer.accumulate_grad_batches

        # Robust fallback for do_not_mask
        if "do_not_mask" not in batch:
            batch["do_not_mask"] = torch.zeros_like(
                batch["input_ids"], dtype=torch.bool
            )

        losses = self._loss(
            x0=batch["input_ids"],
            valid_tokens=batch["attention_mask"],
            do_not_mask=batch["do_not_mask"],
            current_accumulation_step=current_accumulation_step,
            train_mode=True,
            ground_truth_masking=self.config.training.ground_truth_masking,
        )
        self.metrics.update_train(losses.nlls, losses.prior_loss, losses.num_tokens)
        self.log(
            name="trainer/loss",
            value=losses.loss.item(),
            on_step=True,
            on_epoch=False,
            sync_dist=True,
        )
        return losses.loss

    def on_train_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(
                name=k, value=v.compute(), on_step=False, on_epoch=True, sync_dist=True
            )

    def on_validation_epoch_start(self):
        self.metrics.reset()
        self.val_length_metrics.reset()
        self._eval_mode()
        assert self.metrics.valid_nlls.nll.mean_value == 0
        assert self.metrics.valid_nlls.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        # Robust fallback for do_not_mask
        if "do_not_mask" not in batch:
            batch["do_not_mask"] = torch.zeros_like(
                batch["input_ids"], dtype=torch.bool
            )

        losses = self._loss(
            x0=batch["input_ids"],
            valid_tokens=batch["attention_mask"],
            do_not_mask=batch["do_not_mask"],
            train_mode=False,
            ground_truth_masking=self.config.training.ground_truth_masking,
        )
        self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)

        # --- Formal accuracy evaluation ---
        all_generated_samples = dict()
        prompts, targets = self._extract_prompts_and_targets(
            batch["input_ids"], batch["do_not_mask"]
        )

        # Generate completions conditioned on prompts
        # Get evaluation configurations (mode, top_k_fn, suffix combinations)
        eval_configs = self._get_eval_configs()

        # Compute sequence lengths from raw text (without special tokens)
        # This matches the length ranges in config files
        if "text" in batch:
            # For formal language tasks, measure length of INPUT part (before '#')
            # not the full sequence which may include traces/intermediate steps
            lengths_list = []
            for text in batch["text"]:
                if "#" in text:
                    # Use only the input part before the separator
                    input_part = text.split("#")[0].strip()
                    lengths_list.append(len(input_part.split()))
                else:
                    # No separator, use full text
                    lengths_list.append(len(text.strip().split()))

            seq_lengths = torch.tensor(
                lengths_list, dtype=torch.long, device=targets.device
            )
        else:
            # Fallback: count non-padding tokens in targets (includes BOS/EOS)
            # Subtract 2 to approximate raw length
            target_mask = targets != self.tokenizer.pad_token_id
            seq_lengths = target_mask.sum(dim=1) - 2  # (batch_size,)

        for gen_mode, top_k_fn, suffix in eval_configs:
            # Pass the `targets` tensor for shape compatibility, as required by the function signature.
            generated = self.generate_conditioned(
                prompts, targets, mode=gen_mode, top_k_fn=top_k_fn
            )

            # Create display name for logging
            display_mode = f"{gen_mode}{suffix}"

            # Compute accuracy (exact match and token-level)
            acc_exact, acc_token, correct_prediction = self._compute_accuracy(
                generated, targets
            )

            # Also get per-sample metrics for length stratification
            (
                acc_exact_per_sample,
                acc_token_per_sample,
                correct_prediction_per_sample,
            ) = self._compute_accuracy_per_sample(generated, targets)

            # Update length-stratified metrics
            self.val_length_metrics.update(
                mode=display_mode,
                lengths=seq_lengths,
                per_sample_metrics={
                    "acc_exact": acc_exact_per_sample,
                    "acc_token": acc_token_per_sample,
                    "correct_prediction": correct_prediction_per_sample,
                },
            )

            # Use separate W&B panel for each generation method
            self.log(
                f"val_{display_mode}/acc_exact",
                acc_exact,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val_{display_mode}/acc_token",
                acc_token,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"val_{display_mode}/correct_prediction",
                correct_prediction,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            # # Also log top_k_half as default for fair comparison with other models
            # # top_k with half_remaining is the canonical default for MDLM
            # if gen_mode == "top_k" and suffix == "_half" and self.config.algo.name == "mdlm":
            #     self.log("val/default_acc_exact", acc_exact, on_step=False, on_epoch=True, sync_dist=True)
            #     self.log("val/default_acc_token", acc_token, on_step=False, on_epoch=True, sync_dist=True)
            #     self.log("val/default_correct_prediction", correct_prediction, on_step=False, on_epoch=True, sync_dist=True)

            # Log primary accuracy metric for default configurations only
            # For MDLM: top_k_half (top_k with half_remaining)
            # For AR/LT: default mode
            if display_mode in ["default", "top_k_half"]:
                self.log(
                    "val/acc_token",
                    acc_token,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

            # Logic for logging samples remains the same
            if self.trainer.global_rank == 0 and hasattr(
                self.trainer.logger, "log_table"
            ):
                generated_samples = self.tokenizer.batch_decode(
                    generated[: self.config.sampling.num_sample_log],
                    skip_special_tokens=True,
                )
                all_generated_samples[display_mode] = generated_samples

        # Logic for logging samples remains the same
        if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, "log_table"):
            _all_generated_samples = []
            for i in range(self.config.sampling.num_sample_log):
                _all_generated_samples.append(
                    list(
                        all_generated_samples[_gen_mode][i]
                        for _gen_mode in all_generated_samples
                    )
                )

            target_samples = self.tokenizer.batch_decode(
                batch["input_ids"][: self.config.sampling.num_sample_log],
                skip_special_tokens=True,
            )
            self.trainer.logger.log_table(
                key=f"conditioned_generation@global_step{self.global_step}",
                columns=[
                    f"Generated {_gen_mode}" for _gen_mode in all_generated_samples
                ]
                + ["Target"],
                data=[s + [t] for s, t in zip(_all_generated_samples, target_samples)],
            )
        return {"loss": losses.loss, "acc_exact": acc_exact, "acc_token": acc_token}

    def _extract_prompts_and_targets(self, input_ids, do_not_mask):
        """
        Splits input_ids into a prompt tensor and a target tensor using a boolean mask.
        - The prompt tensor contains the original tokens where the mask is True, and padding elsewhere.
        - The target tensor contains the original tokens where the mask is False, and a specific
          ignore_index (-100) elsewhere, which is standard practice for loss functions.
        """

        targets = input_ids.clone()
        targets[do_not_mask] = self.tokenizer.pad_token_id

        prompts = input_ids.clone()
        prompts[~do_not_mask & (input_ids != self.tokenizer.pad_token_id)] = (
            self.tokenizer.mask_token_id
        )

        return prompts, targets

    def _compute_accuracy(self, generated, targets):
        """
        Computes exact match and token-level accuracy.
        """
        # `target_mask` is True only for tokens that should be predicted.
        target_mask = targets != self.tokenizer.pad_token_id

        # 1. Exact Match Accuracy: Percentage of sequences that are perfectly correct.
        # For each sequence, check if all target tokens are correct.
        # A token is considered correct if it matches the generated token OR it's not a target token.
        is_correct_or_ignored = (generated == targets) | ~target_mask
        exact_match = is_correct_or_ignored.all(dim=1)
        acc_exact = exact_match.float().mean().item()

        # 2. Token-level Accuracy: Percentage of all target tokens that are correct.
        # Count correctly predicted tokens within the target mask.
        num_correct_tokens = ((generated == targets) & target_mask).sum().item()
        num_target_tokens = target_mask.sum().item()

        # Avoid division by zero if there are no target tokens in the batch.
        acc_token = (
            num_correct_tokens / num_target_tokens if num_target_tokens > 0 else 0.0
        )

        # 3. Last Prompt Token Accuracy: Is the prediction correct at the last non-padding token in the prompt?
        # Find the last index that is True in target_mask for each sequence.
        last_prompt_indices = target_mask.float().cumsum(dim=1).argmax(dim=1)
        # Clamp indices to valid range
        last_prompt_indices = torch.clamp(last_prompt_indices, 0, targets.shape[1] - 1)
        # Only consider if there is at least one target token
        has_prompt = target_mask.any(dim=1)
        last_prompt_targets = targets[
            torch.arange(targets.shape[0]), last_prompt_indices
        ]
        last_prompt_preds = generated[
            torch.arange(generated.shape[0]), last_prompt_indices
        ]
        last_prompt_correct = (last_prompt_preds == last_prompt_targets) & has_prompt
        correct_prediction = last_prompt_correct.float().mean().item()

        return acc_exact, acc_token, correct_prediction

    def _compute_accuracy_per_sample(self, generated, targets):
        """
        Computes per-sample accuracy metrics (returns tensors, not scalars).
        Used for length-stratified metrics tracking.

        Returns:
            acc_exact: Tensor of shape (batch_size,) with 1.0 for exact matches, 0.0 otherwise
            acc_token: Tensor of shape (batch_size,) with token-level accuracy per sequence
            correct_prediction: Tensor of shape (batch_size,) with 1.0 for correct final predictions
        """
        # `target_mask` is True only for tokens that should be predicted.
        target_mask = targets != self.tokenizer.pad_token_id

        # 1. Exact Match Accuracy per sample
        is_correct_or_ignored = (generated == targets) | ~target_mask
        acc_exact_per_sample = is_correct_or_ignored.all(dim=1).float()  # (batch_size,)

        # 2. Token-level Accuracy per sample
        num_correct_per_sample = (
            ((generated == targets) & target_mask).sum(dim=1).float()
        )
        num_target_per_sample = target_mask.sum(dim=1).float()
        # Avoid division by zero
        acc_token_per_sample = torch.where(
            num_target_per_sample > 0,
            num_correct_per_sample / num_target_per_sample,
            torch.zeros_like(num_correct_per_sample),
        )

        # 3. Last Prompt Token Accuracy per sample
        last_prompt_indices = target_mask.float().cumsum(dim=1).argmax(dim=1)
        last_prompt_indices = torch.clamp(last_prompt_indices, 0, targets.shape[1] - 1)
        has_prompt = target_mask.any(dim=1)
        last_prompt_targets = targets[
            torch.arange(targets.shape[0]), last_prompt_indices
        ]
        last_prompt_preds = generated[
            torch.arange(generated.shape[0]), last_prompt_indices
        ]
        correct_prediction_per_sample = (
            (last_prompt_preds == last_prompt_targets) & has_prompt
        ).float()

        return acc_exact_per_sample, acc_token_per_sample, correct_prediction_per_sample

    def _get_eval_configs(self):
        """
        Get list of (mode, top_k_fn, suffix) tuples for evaluation.

        Returns:
            List of tuples: (mode_name, top_k_fn, display_suffix)
            where top_k_fn can be "half_remaining", an int, or a callable
        """
        if self.config.algo.name == "mdlm":
            # Get constant k value from config
            constant_k = getattr(self.config.eval, "top_k", 1)

            eval_configs = []

            # Random mode always follows MDM masking schedule (no strategy variants)
            eval_configs.append(("random", "half_remaining", ""))

            # Top-k modes with different strategies
            for mode in ["top_k", "top_k_margin"]:
                eval_configs.append((mode, "half_remaining", "_half"))
                eval_configs.append((mode, constant_k, "_const"))

            # Other modes (no strategy variants)
            for mode in ["autoregressive", "one_level", "all_at_once"]:
                eval_configs.append((mode, "half_remaining", ""))

            return eval_configs
        else:
            return [("default", "half_remaining", "")]

    def generate_conditioned(
        self, prompts, targets=None, mode="random", top_k_fn="half_remaining"
    ):
        """
        Generate completions conditioned on prompts.

        Args:
            prompts: (batch, seq) tensor with masked positions
            targets: (batch, seq) tensor (optional, used by some algorithms for shape/structure)
            mode: Generation mode (varies by algorithm)
            top_k_fn: Strategy for determining k (number of positions to unmask per step).
                      Options:
                      - "half_remaining" (default): k = max(1, remaining // 2)
                      - int: constant k value
                      - callable: custom function taking state dict and returning k

        Returns:
            (batch, seq) tensor of generated completions
        """
        # Stub: implement in subclass or algo
        raise NotImplementedError(
            "Implement prompt-conditioned generation with unmasking modes in subclass/algo."
        )

    def on_validation_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(
                name=k, value=v.compute(), on_step=False, on_epoch=True, sync_dist=True
            )

        # Compute and log length-stratified metrics
        length_stratified_logs = self.val_length_metrics.get_wandb_logs(prefix="val")
        for metric_name, metric_value in length_stratified_logs.items():
            self.log(
                name=metric_name,
                value=metric_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # Save validation metrics to file
        val_metrics_file = os.path.join(
            self.config.checkpointing.save_dir, "validation_metrics.json"
        )
        current_metrics = {
            k: v.compute().item() for k, v in self.metrics.valid_nlls.items()
        }
        current_metrics["epoch"] = self.current_epoch
        current_metrics["global_step"] = self.global_step

        # Also save all logged metrics (including decoding strategy metrics)
        # Get all callback metrics that were logged this epoch
        if hasattr(self.trainer, "callback_metrics"):
            for metric_name, metric_value in self.trainer.callback_metrics.items():
                if metric_name not in current_metrics and metric_name not in [
                    "epoch",
                    "global_step",
                ]:
                    # Only save validation metrics and avoid duplicates
                    if isinstance(metric_value, torch.Tensor):
                        current_metrics[metric_name] = metric_value.item()
                    elif isinstance(metric_value, (int, float)):
                        current_metrics[metric_name] = metric_value

        # Add length-stratified metrics to saved file
        current_metrics["length_stratified"] = self.val_length_metrics.compute_all()

        if os.path.exists(val_metrics_file):
            with open(val_metrics_file, "r") as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []

        all_metrics.append(current_metrics)

        with open(val_metrics_file, "w") as f:
            json.dump(all_metrics, f, indent=4)

        # if (
        #     self.config.eval.compute_perplexity_on_sanity
        #     or not self.trainer.sanity_checking
        # ) and self.config.eval.generate_samples:
        #     samples, text_samples = None, None
        #     for _ in range(self.config.sampling.num_sample_batches):
        #         samples = self.generate_samples(
        #             num_samples=self.config.loader.eval_batch_size
        #         )

        #         self.metrics.record_entropy(samples)
        #         # Decode the samples to be re-tokenized by eval model
        #         text_samples = self.tokenizer.batch_decode(samples)
        #         if self.config.eval.compute_generative_perplexity:
        #             self.metrics.record_generative_perplexity(
        #                 text_samples, self.num_tokens, self.device
        #             )
        #     if text_samples is not None:
        #         if self.trainer.global_rank == 0 and hasattr(
        #             self.trainer.logger, "log_table"
        #         ):
        #             # Log the last generated samples[: self.config.sampling.num_sample_log]
        #             text_samples = text_samples
        #             self.trainer.logger.log_table(
        #                 key=f"samples@global_step{self.global_step}",
        #                 columns=["Generated Samples"],
        #                 data=[[s] for s in text_samples],
        #             )
        #         if self.config.eval.compute_generative_perplexity:
        #             self.log(
        #                 "val/gen_ppl",
        #                 self.metrics.gen_ppl.compute(),
        #                 on_epoch=True,
        #                 on_step=False,
        #                 sync_dist=True,
        #             )
        #             self.log(
        #                 "val/sample_entropy",
        #                 self.metrics.sample_entropy.compute(),
        #                 on_epoch=True,
        #                 on_step=False,
        #                 sync_dist=True,
        #             )
        self._train_mode()

    def on_test_epoch_start(self):
        self.metrics.reset()
        self.test_length_metrics.reset()
        self._eval_mode()
        assert self.metrics.valid_nlls.nll.mean_value == 0
        assert self.metrics.valid_nlls.nll.weight == 0

    def test_step(self, batch, batch_idx):
        # Robust fallback for do_not_mask
        if "do_not_mask" not in batch:
            batch["do_not_mask"] = torch.zeros_like(
                batch["input_ids"], dtype=torch.bool
            )

        losses = self._loss(
            x0=batch["input_ids"],
            valid_tokens=batch["attention_mask"],
            do_not_mask=batch["do_not_mask"],
            train_mode=False,
            ground_truth_masking=self.config.training.ground_truth_masking,
        )
        self.metrics.update_valid(losses.nlls, losses.prior_loss, losses.num_tokens)

        # --- Formal accuracy evaluation ---
        all_generated_samples = dict()
        prompts, targets = self._extract_prompts_and_targets(
            batch["input_ids"], batch["do_not_mask"]
        )

        # Generate completions conditioned on prompts
        # Get evaluation configurations (mode, top_k_fn, suffix combinations)
        eval_configs = self._get_eval_configs()

        # Compute sequence lengths from raw text (without special tokens)
        # This matches the length ranges in config files
        if "text" in batch:
            # For formal language tasks, measure length of INPUT part (before '#')
            # not the full sequence which may include traces/intermediate steps
            lengths_list = []
            for text in batch["text"]:
                if "#" in text:
                    # Use only the input part before the separator
                    input_part = text.split("#")[0].strip()
                    lengths_list.append(len(input_part.split()))
                else:
                    # No separator, use full text
                    lengths_list.append(len(text.strip().split()))

            seq_lengths = torch.tensor(
                lengths_list, dtype=torch.long, device=targets.device
            )
        else:
            # Fallback: count non-padding tokens in targets (includes BOS/EOS)
            # Subtract 2 to approximate raw length
            target_mask = targets != self.tokenizer.pad_token_id
            seq_lengths = target_mask.sum(dim=1) - 2  # (batch_size,)

        for gen_mode, top_k_fn, suffix in eval_configs:
            # Pass the `targets` tensor for shape compatibility, as required by the function signature.
            generated = self.generate_conditioned(
                prompts, targets, mode=gen_mode, top_k_fn=top_k_fn
            )

            # Create display name for logging
            display_mode = f"{gen_mode}{suffix}"

            # Compute accuracy (exact match and token-level)
            acc_exact, acc_token, correct_prediction = self._compute_accuracy(
                generated, targets
            )

            # Also get per-sample metrics for length stratification
            (
                acc_exact_per_sample,
                acc_token_per_sample,
                correct_prediction_per_sample,
            ) = self._compute_accuracy_per_sample(generated, targets)

            # Update length-stratified metrics
            self.test_length_metrics.update(
                mode=display_mode,
                lengths=seq_lengths,
                per_sample_metrics={
                    "acc_exact": acc_exact_per_sample,
                    "acc_token": acc_token_per_sample,
                    "correct_prediction": correct_prediction_per_sample,
                },
            )

            # Use separate W&B panel for each generation method
            self.log(
                f"test_{display_mode}/acc_exact",
                acc_exact,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"test_{display_mode}/acc_token",
                acc_token,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                f"test_{display_mode}/correct_prediction",
                correct_prediction,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

            # Also log top_k_half as default for fair comparison with other models
            # top_k with half_remaining is the canonical default for MDLM
            if (
                gen_mode == "top_k"
                and suffix == "_half"
                and self.config.algo.name == "mdlm"
            ):
                self.log(
                    "test/default_acc_exact",
                    acc_exact,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test/default_acc_token",
                    acc_token,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    "test/default_correct_prediction",
                    correct_prediction,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

            # Log primary accuracy metric for default configurations only
            # For MDLM: top_k_half (top_k with half_remaining)
            # For AR/LT: default mode
            if display_mode in ["default", "top_k_half"]:
                self.log(
                    "test/acc_token",
                    acc_token,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

            # Logic for logging samples remains the same
            if self.trainer.global_rank == 0 and hasattr(
                self.trainer.logger, "log_table"
            ):
                generated_samples = self.tokenizer.batch_decode(
                    generated[: self.config.sampling.num_sample_log],
                    skip_special_tokens=True,
                )
                all_generated_samples[display_mode] = generated_samples

        # Logic for logging samples remains the same
        if self.trainer.global_rank == 0 and hasattr(self.trainer.logger, "log_table"):
            _all_generated_samples = []
            for i in range(self.config.sampling.num_sample_log):
                _all_generated_samples.append(
                    list(
                        all_generated_samples[_gen_mode][i]
                        for _gen_mode in all_generated_samples
                    )
                )

            target_samples = self.tokenizer.batch_decode(
                batch["input_ids"][: self.config.sampling.num_sample_log],
                skip_special_tokens=True,
            )
            self.trainer.logger.log_table(
                key=f"test_conditioned_generation@global_step{self.global_step}",
                columns=[
                    f"Generated {_gen_mode}" for _gen_mode in all_generated_samples
                ]
                + ["Target"],
                data=[s + [t] for s, t in zip(_all_generated_samples, target_samples)],
            )
        return {"loss": losses.loss, "acc_exact": acc_exact, "acc_token": acc_token}

    def on_test_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(
                name="test/" + k,
                value=v.compute(),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # Compute and log length-stratified metrics
        length_stratified_logs = self.test_length_metrics.get_wandb_logs(prefix="test")
        for metric_name, metric_value in length_stratified_logs.items():
            self.log(
                name=metric_name,
                value=metric_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        # Save test metrics to file
        test_metrics_file = os.path.join(
            self.config.checkpointing.save_dir, "test_metrics.json"
        )
        current_metrics = {
            "test/" + k: v.compute().item() for k, v in self.metrics.valid_nlls.items()
        }
        current_metrics["epoch"] = self.current_epoch
        current_metrics["global_step"] = self.global_step

        # Also save all logged metrics (including decoding strategy metrics)
        if hasattr(self.trainer, "callback_metrics"):
            for metric_name, metric_value in self.trainer.callback_metrics.items():
                if metric_name not in current_metrics and metric_name not in [
                    "epoch",
                    "global_step",
                ]:
                    if isinstance(metric_value, torch.Tensor):
                        current_metrics[metric_name] = metric_value.item()
                    elif isinstance(metric_value, (int, float)):
                        current_metrics[metric_name] = metric_value

        # Add length-stratified metrics to saved file
        current_metrics["length_stratified"] = self.test_length_metrics.compute_all()

        with open(test_metrics_file, "w") as f:
            json.dump(current_metrics, f, indent=4)

        self._train_mode()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._get_parameters(),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/loss",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    def generate_samples(self, num_samples, num_steps, eps):
        raise NotImplementedError

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        self._eval_mode()
        samples = self.generate_samples(
            num_samples=self.config.loader.eval_batch_size, num_steps=num_steps, eps=eps
        )
        self._train_mode()
        return samples

    def _process_model_input(self, x0, valid_tokens):
        raise NotImplementedError

    def nll(
        self,
        input_tokens,
        output_tokens,
        do_not_mask,
        current_accumulation_step=None,
        train_mode=False,
        ground_truth_masking=False,
    ):
        raise NotImplementedError

    def _loss(
        self,
        x0,
        valid_tokens,
        do_not_mask,
        current_accumulation_step=None,
        train_mode=False,
        ground_truth_masking=False,
    ):
        # TODO: Use valid_tokens instead of do_not_mask?
        (input_tokens, output_tokens, valid_tokens) = self._process_model_input(
            x0, valid_tokens
        )

        loss = self.nll(
            input_tokens,
            output_tokens,
            do_not_mask,
            current_accumulation_step,
            train_mode,
            ground_truth_masking,
        )
        assert loss.ndim == 2

        if output_tokens is not None:  # LT or AR case
            nlls = loss.sum()
            num_tokens = (input_tokens == self.tokenizer.mask_token_id).sum()
            if num_tokens == 0:
                # Find the location of the first '#' and last '|' in each sequence
                hash_token_id = self.tokenizer.convert_tokens_to_ids("#")
                pipe_token_id = self.tokenizer.convert_tokens_to_ids("|")
                # If not found, fallback to hack
                if (
                    hash_token_id == self.tokenizer.unk_token_id
                    or pipe_token_id == self.tokenizer.unk_token_id
                ):
                    num_tokens = len(loss)  # fallback hack
                else:
                    # Compute per sequence
                    first_hash = (input_tokens == hash_token_id).float().argmax(dim=1)
                    last_pipe = (input_tokens == pipe_token_id).float().cumsum(dim=1)
                    last_pipe = (
                        (last_pipe == last_pipe.max(dim=1, keepdim=True)[0])
                        .float()
                        .argmax(dim=1)
                    )
                    # Clamp to valid range
                    num_tokens = (last_pipe - first_hash).clamp(min=0).sum().item()
            token_nll = nlls / num_tokens
        else:  # MDM case
            nlls = (loss * valid_tokens).sum()
            num_tokens = valid_tokens.sum()
            token_nll = nlls / num_tokens

        return Loss(loss=token_nll, nlls=nlls, prior_loss=0.0, num_tokens=num_tokens)
