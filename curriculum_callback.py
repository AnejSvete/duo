"""
Curriculum Learning Callback for Progressive Length Training

This callback implements curriculum learning by progressively training on longer sequences.
The training data is divided into bins based on sequence length, and the model trains on
each bin for a specified number of epochs before moving to the next bin.
"""

import logging
from typing import Optional

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback

LOGGER = logging.getLogger(__name__)


class CurriculumLearningCallback(Callback):
    """
    Implements curriculum learning by progressively training on longer sequences.

    The callback divides the training length range into bins and trains on each bin
    for a specified number of epochs. Bins can optionally overlap to ensure smooth
    transitions.

    Args:
        enabled: Whether curriculum learning is enabled
        num_bins: Number of length bins to divide training into
        epochs_per_bin: Number of epochs to train on each bin
        overlap: Overlap ratio between consecutive bins (0-1)
        min_train_len: Minimum training sequence length
        max_train_len: Maximum training sequence length
    """

    def __init__(
        self,
        enabled: bool = False,
        num_bins: int = 4,
        epochs_per_bin: int = 5,
        overlap: float = 0.2,
        min_train_len: Optional[int] = None,
        max_train_len: Optional[int] = None,
    ):
        super().__init__()
        self.enabled = enabled
        self.num_bins = num_bins
        self.epochs_per_bin = epochs_per_bin
        self.overlap = overlap
        self.min_train_len = min_train_len
        self.max_train_len = max_train_len

        self.current_bin = 0
        self.epochs_in_current_bin = 0
        self.bin_boundaries = None

        if not enabled:
            LOGGER.info("Curriculum learning is disabled")

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        """Setup the curriculum learning boundaries."""
        if not self.enabled or stage != "fit":
            return

        # Get length range from config if not provided
        if self.min_train_len is None or self.max_train_len is None:
            config = pl_module.config
            self.min_train_len = config.data.properties.min_train_len
            self.max_train_len = config.data.properties.max_train_len

        # Calculate bin boundaries with overlap
        self._compute_bin_boundaries()

        LOGGER.info(f"Curriculum Learning enabled with {self.num_bins} bins")
        LOGGER.info(f"Length range: [{self.min_train_len}, {self.max_train_len}]")
        LOGGER.info(f"Epochs per bin: {self.epochs_per_bin}")
        LOGGER.info(f"Overlap ratio: {self.overlap}")
        LOGGER.info(f"Bin boundaries: {self.bin_boundaries}")

    def _compute_bin_boundaries(self) -> None:
        """Compute the length boundaries for each bin with optional overlap."""
        total_range = self.max_train_len - self.min_train_len

        if self.num_bins == 1:
            # Single bin covers entire range
            self.bin_boundaries = [(self.min_train_len, self.max_train_len)]
            return

        # Calculate bin size without overlap
        base_bin_size = total_range / self.num_bins
        overlap_size = base_bin_size * self.overlap

        self.bin_boundaries = []
        for i in range(self.num_bins):
            # Calculate bin start and end with overlap
            bin_start = self.min_train_len + i * base_bin_size
            bin_end = bin_start + base_bin_size + overlap_size

            # Clamp to valid range
            bin_start = max(self.min_train_len, int(bin_start))
            bin_end = min(self.max_train_len, int(bin_end))

            # Ensure last bin reaches max length
            if i == self.num_bins - 1:
                bin_end = self.max_train_len

            self.bin_boundaries.append((bin_start, bin_end))

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Check if we should move to the next bin at the start of each epoch."""
        if not self.enabled:
            return

        # Check if we should advance to next bin
        if self.epochs_in_current_bin >= self.epochs_per_bin and self.current_bin < self.num_bins - 1:
            self.current_bin += 1
            self.epochs_in_current_bin = 0
            LOGGER.info(f"Advancing to curriculum bin {self.current_bin + 1}/{self.num_bins}")

        # Get current bin boundaries
        min_len, max_len = self.bin_boundaries[self.current_bin]

        LOGGER.info(
            f"Epoch {trainer.current_epoch}: Training on length range [{min_len}, {max_len}] "
            f"(bin {self.current_bin + 1}/{self.num_bins}, epoch {self.epochs_in_current_bin + 1}/{self.epochs_per_bin})"
        )

        # Apply length filter to dataloader
        self._apply_length_filter(trainer, min_len, max_len)

        self.epochs_in_current_bin += 1

    def _apply_length_filter(
        self,
        trainer: L.Trainer,
        min_len: int,
        max_len: int
    ) -> None:
        """
        Apply length filtering to the training dataloader.

        This creates a filtered view of the dataset that only includes examples
        within the specified length range.
        """
        train_dataloader = trainer.train_dataloader
        if train_dataloader is None:
            return

        # Get the underlying dataset
        dataset = train_dataloader.dataset

        # Create indices for examples within the length range
        filtered_indices = []
        for idx in range(len(dataset)):
            # Get the sequence (without BOS/EOS tokens for length calculation)
            example = dataset[idx]

            # Calculate actual sequence length by counting non-padding tokens
            if 'attention_mask' in example:
                seq_len = example['attention_mask'].sum().item()
            elif 'input_ids' in example:
                # Count tokens that are not padding
                tokenizer = train_dataloader.tokenizer
                input_ids = example['input_ids']
                seq_len = (input_ids != tokenizer.pad_token_id).sum().item()
            else:
                continue

            # Include example if within current bin range
            if min_len <= seq_len <= max_len:
                filtered_indices.append(idx)

        LOGGER.info(
            f"Filtered dataset: {len(filtered_indices)}/{len(dataset)} examples "
            f"in length range [{min_len}, {max_len}]"
        )

        # Create a subset of the dataset
        if len(filtered_indices) > 0:
            filtered_dataset = torch.utils.data.Subset(dataset, filtered_indices)

            # Create new dataloader with filtered dataset
            new_dataloader = torch.utils.data.DataLoader(
                filtered_dataset,
                batch_size=train_dataloader.batch_size,
                num_workers=train_dataloader.num_workers,
                pin_memory=train_dataloader.pin_memory,
                shuffle=True,
                persistent_workers=train_dataloader.persistent_workers,
                collate_fn=train_dataloader.collate_fn,
            )
            new_dataloader.tokenizer = train_dataloader.tokenizer

            # Replace the trainer's dataloader
            trainer.train_dataloader = new_dataloader
        else:
            LOGGER.warning(
                f"No examples found in length range [{min_len}, {max_len}]. "
                "Keeping full dataset."
            )

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log curriculum learning completion."""
        if self.enabled:
            LOGGER.info(
                f"Curriculum learning completed. Trained through {self.current_bin + 1}/{self.num_bins} bins."
            )

    def state_dict(self):
        """Save callback state for checkpointing."""
        return {
            'current_bin': self.current_bin,
            'epochs_in_current_bin': self.epochs_in_current_bin,
            'bin_boundaries': self.bin_boundaries,
        }

    def load_state_dict(self, state_dict):
        """Load callback state from checkpoint."""
        self.current_bin = state_dict['current_bin']
        self.epochs_in_current_bin = state_dict['epochs_in_current_bin']
        self.bin_boundaries = state_dict['bin_boundaries']
        LOGGER.info(
            f"Resumed curriculum learning at bin {self.current_bin + 1}/{self.num_bins}, "
            f"epoch {self.epochs_in_current_bin + 1}/{self.epochs_per_bin}"
        )
