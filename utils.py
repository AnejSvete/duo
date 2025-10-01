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
