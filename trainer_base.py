import itertools
import math
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import transformers

import dataloader
import metrics
import models
import utils

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
    def __init__(self):
        super().__init__()
        self.eps = 1e-3  # To be consistent with SEDD: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/0605786da5ccb5747545e26d66fdf477187598b6/noise_lib.py#L56

    def forward(self, t):
        t = (1 - self.eps) * t
        alpha_t = 1 - t
        dalpha_t = -(1 - self.eps)
        return dalpha_t, alpha_t


def sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


class TrainerBase(L.LightningModule):
    def __init__(
        self, config, tokenizer: transformers.PreTrainedTokenizer, vocab_size=None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        if hasattr(self.config.algo, "ignore_bos"):
            self.ignore_bos = config.algo.ignore_bos
        else:
            self.ignore_bos = False
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
                loop_depth_function = lambda n: math.ceil(math.log2(n) / 2)
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
        elif self.config.algo.backbone == "dimamba":
            self.backbone = models.dimamba.DiMamba(
                self.config,
                vocab_size=self.vocab_size,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elif self.config.algo.backbone == "hf_dit":
            self.backbone = transformers.AutoModelForMaskedLM.from_pretrained(
                config.eval.checkpoint_path, trust_remote_code=True
            )

        self.T = self.config.algo.T
        self.num_tokens = self.config.model.length
        self.softplus = torch.nn.Softplus()
        self.p_nucleus = self.config.sampling.p_nucleus
        # Noise Schedule
        self.noise = LogLinear()

        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=self.config.eval.perplexity_batch_size,
        )

        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                self._get_parameters(), decay=self.config.training.ema
            )
        else:
            self.ema = None

        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.algo.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None

    def _validate_configuration(self):
        assert self.config.algo.backbone in {"dit", "hf_dit", "lt"}
        if self.config.algo.parameterization == "ar":
            assert not self.config.algo.time_conditioning
            assert self.config.prior.type == "none"

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
        if self.ema:
            self.ema.store(self._get_parameters())
            self.ema.copy_to(self._get_parameters())
        self.backbone.eval()
        self.noise.eval()

    def _train_mode(self):
        if self.ema:
            self.ema.restore(self._get_parameters())
        self.backbone.train()
        self.noise.train()

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        # self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"][
        #     "current"
        # ]["completed"]
        # self.fast_forward_batches = checkpoint["loops"]["fit_loop"][
        #     "epoch_loop.batch_progress"
        # ]["current"]["completed"]

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed']
        # is 1 iteration behind, so we're using the optimizer's progress.
        # checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
        #     "completed"
        # ] = (
        #     checkpoint["loops"]["fit_loop"][
        #         "epoch_loop.automatic_optimization.optim_progress"
        #     ]["optimizer"]["step"]["total"]["completed"]
        #     * self.trainer.accumulate_grad_batches
        # )
        # checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
        #     "completed"
        # ] = (
        #     checkpoint["loops"]["fit_loop"][
        #         "epoch_loop.automatic_optimization.optim_progress"
        #     ]["optimizer"]["step"]["current"]["completed"]
        #     * self.trainer.accumulate_grad_batches
        # )
        # # _batches_that_stepped tracks the number of global steps,
        # # not the number of local steps, so we don't multiply with
        # # self.trainer.accumulate_grad_batches here.
        # checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"][
        #     "_batches_that_stepped"
        # ] = checkpoint["loops"]["fit_loop"][
        #     "epoch_loop.automatic_optimization.optim_progress"
        # ][
        #     "optimizer"
        # ][
        #     "step"
        # ][
        #     "total"
        # ][
        #     "completed"
        # ]
        # if "sampler" not in checkpoint.keys():
        #     checkpoint["sampler"] = {}
        # if hasattr(self.trainer.train_dataloader.sampler, "state_dict"):
        #     sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
        #     checkpoint["sampler"]["random_state"] = sampler_state_dict.get(
        #         "random_state", None
        #     )
        # else:
        #     checkpoint["sampler"]["random_state"] = None

    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self._get_parameters())

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
        # Only run if formal dataset (do_not_mask is used)
        if batch["do_not_mask"].any():
            all_generated_samples = dict()
            prompts, targets = self._extract_prompts_and_targets(
                batch["input_ids"], batch["do_not_mask"]
            )

            # Generate completions conditioned on prompts
            # for gen_mode in ["random", "top_k", "one_level", "all_at_once"]:
            top_k = getattr(self.config.eval, "top_k", 1)
            for gen_mode in ["one_level"]:
                # Pass the `targets` tensor for shape compatibility, as required by the function signature.
                generated = self.generate_conditioned(
                    prompts, targets, mode=gen_mode, top_k=top_k
                )

                # Compute accuracy (exact match and token-level)
                acc_exact, acc_token, correct_prediction = self._compute_accuracy(
                    generated, targets
                )
                self.log(
                    f"val/{gen_mode}_acc_exact",
                    acc_exact,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    f"val/{gen_mode}_acc_token",
                    acc_token,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    f"val/{gen_mode}_correct_prediction",
                    correct_prediction,
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
                    all_generated_samples[gen_mode] = generated_samples

            # Logic for logging samples remains the same
            if self.trainer.global_rank == 0 and hasattr(
                self.trainer.logger, "log_table"
            ):
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
                    data=[
                        s + [t] for s, t in zip(_all_generated_samples, target_samples)
                    ],
                )
            return {"loss": losses.loss, "acc_exact": acc_exact, "acc_token": acc_token}

        return losses.loss

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

        print("-------------------------")
        print(f"generated: {generated[:4]}")
        print(f"targets: {targets[:4]}")
        print(f"target_mask: {target_mask[:4]}")
        print("-------------------------")

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

    def generate_conditioned(self, prompts, mode="random", top_k=1):
        # Stub: implement in subclass or algo
        # prompts: (batch, seq) tensor
        # Return: (batch, seq) tensor of generated completions (same length as targets)
        raise NotImplementedError(
            "Implement prompt-conditioned generation with unmasking modes in subclass/algo."
        )

    def on_validation_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(
                name=k, value=v.compute(), on_step=False, on_epoch=True, sync_dist=True
            )
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
        if self.ignore_bos:
            loss[:, 1:] = loss[:, 1:]
            valid_tokens[:, 1:] = valid_tokens[:, 1:]

        nlls = (loss * valid_tokens).sum()
        num_tokens = valid_tokens.sum()
        token_nll = nlls / num_tokens
        print()
        print()
        print()
        print(f"nlls: {nlls}")
        print(f"token_nll: {token_nll}")
        print(f"num_tokens: {num_tokens}")
        print(f"valid_tokens: {valid_tokens[:4]}")
        print()
        print()
        print()
        return Loss(loss=token_nll, nlls=nlls, prior_loss=0.0, num_tokens=num_tokens)


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

        log_x_theta = self.forward(xt, sigma=sigma)
        # utils.print_nans(log_x_theta, "model_output")  # Assuming utils is available
        return self.nll_per_token(
            log_x_theta=log_x_theta,
            xt=xt,
            x0=x0,
            alpha_t=alpha_t_unsqueezed,
            dalpha_t=dalpha_t,
            low_var=train_mode and self.loss_type == "low_var",
        )

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
        self.subs_masking = config.algo.subs_masking
        super().__init__(config, tokenizer, vocab_size=vocab_size)
        self.save_hyperparameters()

    def _validate_configuration(self):
        super()._validate_configuration()
        if self.parameterization in {"score", "mean"}:
            assert self.time_conditioning
        assert not (self.parameterization == "mean" and self.T == 0)
        if self.T > 0:
            assert self.parameterization in {"mean", "subs"}
        if self.subs_masking:
            assert self.parameterization == "mean"

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

            if self.ignore_bos:
                xt[:, 0] = x[:, 0]
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

            if self.ignore_bos:
                xt[:, 0] = x[:, 0]

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
        if self.config.algo.name != "distillation":
            assert self.T == 0

    def q_xt(self, x, alpha_t, do_not_mask, mask_mode="random"):
        """Computes the noisy sample xt, protecting specified tokens."""
        # Decide which tokens to potentially corrupt based on the noise schedule
        potential_corruption = torch.rand(*x.shape, device=x.device) < 1 - alpha_t

        # Only corrupt tokens where potential_corruption is True AND do_not_mask is False
        final_corruption = potential_corruption & ~do_not_mask

        uniform_tensor = torch.randint(0, self.vocab_size, x.shape, device=x.device)

        xt = torch.where(final_corruption, uniform_tensor, x)
        if self.ignore_bos:
            xt[:, 0] = x[:, 0]
        return xt

    def prior_sample(self, *batch_dims):
        return torch.randint(
            0, self.vocab_size, batch_dims, dtype=torch.int64, device=self.device
        )
