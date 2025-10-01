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
        # Noise Schedule
        self.noise = LogLinear()

        self.metrics = metrics.Metrics(
            gen_ppl_eval_model_name_or_path=self.config.eval.gen_ppl_eval_model_name_or_path,
            eval_ppl_batch_size=self.config.eval.perplexity_batch_size,
        )

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
            top_k = getattr(self.config.eval, "top_k", 1)

            gen_modes = (
                ["random", "top_k", "one_level", "all_at_once", "one_at_a_time"]
                if self.config.algo.name != "ar"
                else ["default"]
            )

            for gen_mode in gen_modes:
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

                if gen_mode in ["default", "one_at_a_time"]:
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

        # Save validation metrics to file
        val_metrics_file = os.path.join(
            self.config.checkpointing.save_dir, "validation_metrics.json"
        )
        current_metrics = {
            k: v.compute().item() for k, v in self.metrics.valid_nlls.items()
        }
        current_metrics["epoch"] = self.current_epoch
        current_metrics["global_step"] = self.global_step

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
        # Only run if formal dataset (do_not_mask is used)
        if batch["do_not_mask"].any():
            all_generated_samples = dict()
            prompts, targets = self._extract_prompts_and_targets(
                batch["input_ids"], batch["do_not_mask"]
            )

            # Generate completions conditioned on prompts
            top_k = getattr(self.config.eval, "top_k", 1)

            gen_modes = (
                ["random", "top_k", "one_level", "all_at_once", "one_at_a_time"]
                if self.config.algo.name != "ar"
                else ["default"]
            )

            for gen_mode in gen_modes:
                # Pass the `targets` tensor for shape compatibility, as required by the function signature.
                generated = self.generate_conditioned(
                    prompts, targets, mode=gen_mode, top_k=top_k
                )

                # Compute accuracy (exact match and token-level)
                acc_exact, acc_token, correct_prediction = self._compute_accuracy(
                    generated, targets
                )
                self.log(
                    f"test/{gen_mode}_acc_exact",
                    acc_exact,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    f"test/{gen_mode}_acc_token",
                    acc_token,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
                self.log(
                    f"test/{gen_mode}_correct_prediction",
                    correct_prediction,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

                if gen_mode in ["default", "one_at_a_time"]:
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
                    key=f"test_conditioned_generation@global_step{self.global_step}",
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

    def on_test_epoch_end(self):
        for k, v in self.metrics.valid_nlls.items():
            self.log(
                name="test/" + k,
                value=v.compute(),
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
