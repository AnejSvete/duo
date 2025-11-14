"""Console logger utilities.

Copied from https://github.com/HazyResearch/transformers/blob/master/src/utils/utils.py
Copied from https://docs.python.org/3/howto/logging-cookbook.html#using-a-context-manager-for-selective-logging
"""

import logging

import fsspec
import lightning
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fsspec_exists(filename):
    """Check if a file exists using fsspec."""
    fs, _ = fsspec.core.url_to_fs(filename)
    return fs.exists(filename)


def fsspec_listdir(dirname):
    """Listdir in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    return fs.ls(dirname)


def fsspec_mkdirs(dirname, exist_ok=True):
    """Mkdirs in manner compatible with fsspec."""
    fs, _ = fsspec.core.url_to_fs(dirname)
    fs.makedirs(dirname, exist_ok=exist_ok)


def print_nans(tensor, name):
    if torch.isnan(tensor).any():
        print(name, tensor)


class LRHalveScheduler:
    def __init__(self, warmup_steps, n_halve_steps):
        self.warmup_steps = warmup_steps
        self.n_halve_steps = n_halve_steps

    def __call__(self, current_step):
        if current_step < self.warmup_steps:
            return current_step / self.warmup_steps
        return 0.5 ** ((current_step - self.warmup_steps) // self.n_halve_steps)


class WarmupStableDecayScheduler:
    """
    Three-phase learning rate schedule:
    1. Warmup: Linear increase from 0 to 1
    2. Stable: Constant at 1
    3. Decay: Cosine decay from 1 to min_lr_ratio

    Args:
        warmup_steps: Number of steps for warmup phase
        stable_steps: Number of steps to maintain peak learning rate
        decay_steps: Number of steps for cosine decay phase
        min_lr_ratio: Minimum learning rate as ratio of base lr (default: 0.1)
    """
    def __init__(self, warmup_steps, stable_steps, decay_steps, min_lr_ratio=0.1):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.min_lr_ratio = min_lr_ratio
        self.stable_end = warmup_steps + stable_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps

    def __call__(self, current_step):
        if current_step < self.warmup_steps:
            # Warmup phase: linear increase
            return current_step / self.warmup_steps
        elif current_step < self.stable_end:
            # Stable phase: constant at peak
            return 1.0
        elif current_step < self.total_steps:
            # Decay phase: cosine decay
            progress = (current_step - self.stable_end) / self.decay_steps
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265359)))
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay
        else:
            # After decay: maintain minimum
            return self.min_lr_ratio


class ReduceLROnPlateauCallback(lightning.Callback):
    """
    Custom callback to reduce learning rate when a metric has stopped improving.
    Works alongside the base scheduler without interfering.

    Args:
        monitor: Metric to monitor (e.g., 'val/loss')
        patience: Number of validation checks with no improvement before reducing LR
        factor: Factor by which to reduce the learning rate (new_lr = lr * factor)
        min_lr: Minimum learning rate threshold
        mode: 'min' for loss, 'max' for accuracy
        verbose: Whether to print messages when LR is reduced
        cooldown: Number of epochs to wait before resuming normal operation after lr reduction
    """
    def __init__(
        self,
        monitor="val/loss",
        patience=4,
        factor=0.5,
        min_lr=1e-6,
        mode="min",
        verbose=True,
        cooldown=2,
    ):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode
        self.verbose = verbose
        self.cooldown = cooldown

        self.wait = 0
        self.cooldown_counter = 0
        self.best = None
        self.mode_worse = torch.inf if mode == "min" else -torch.inf

    def _is_improvement(self, current, best):
        if self.mode == "min":
            return current < best
        else:
            return current > best

    def on_validation_end(self, trainer, pl_module):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            return

        if self.best is None:
            self.best = current
            return

        if self._is_improvement(current, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            # Reduce learning rate for all optimizers and param groups
            for optimizer in trainer.optimizers:
                for param_group in optimizer.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * self.factor, self.min_lr)
                    if new_lr < old_lr:
                        param_group['lr'] = new_lr
                        if self.verbose:
                            print(
                                f"\nReducing learning rate from {old_lr:.2e} to {new_lr:.2e} "
                                f"due to {self.monitor} plateau (patience={self.patience})"
                            )

            self.wait = 0
            self.cooldown_counter = self.cooldown


class GradientInspectionCallback(lightning.Callback):
    def __init__(self, num_grads_log):
        self.num_grads_log = 10

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        gradients = []
        for name, param in pl_module.backbone.blocks.named_parameters():
            gradients.append(param.grad.view(-1))

        if gradients:
            grads = torch.cat((gradients))
            if not hasattr(pl_module, "grad_accum_buffer"):
                pl_module.grad_step = torch.tensor(0, device=pl_module.device)
                pl_module.grad_accum_buffer = torch.zeros(
                    self.num_grads_log, grads.shape[0], device=pl_module.device
                )
            pl_module.grad_accum_buffer[pl_module.grad_step] = grads
            pl_module.grad_step += 1

        if (
            hasattr(pl_module, "grad_accum_buffer")
            and pl_module.grad_step == self.num_grads_log
        ):
            grads = pl_module.grad_accum_buffer
            grad_var = grads.std(0).mean()
            pl_module.log(
                name="trainer/grad_var",
                value=grad_var.item(),
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )
            # import ipdb; ipdb.set_trace()
            # should save the grads tensor as a numpy array
            # and visualize mean, median, top-k
            pl_module.grad_accum_buffer.zero_()
            pl_module.grad_step = 0


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(
            logger,
            level,
            lightning.pytorch.utilities.rank_zero_only(getattr(logger, level)),
        )

    return logger
