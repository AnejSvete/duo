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
        use_quantiles: Use data-driven quantile bins instead of uniform length bins
        sample_size: Number of examples to sample when estimating length distribution
        sample_strategy: Sampling strategy ('linspace' or 'random')
        exact_percentiles: Use exact full-dataset scan for percentile computation
        min_batches_per_epoch: Minimum batches required per epoch (auto-expands range if needed)
        min_examples_per_bin: Minimum examples required per bin (bins with fewer are skipped)
    """

    def __init__(
        self,
        enabled: bool = False,
        num_bins: int = 4,
        epochs_per_bin: int = 5,
        overlap: float = 0.2,
        min_train_len: Optional[int] = None,
        max_train_len: Optional[int] = None,
        use_quantiles: bool = True,
        sample_size: int = 1000,
        sample_strategy: str = "linspace",
        exact_percentiles: bool = True,
        min_batches_per_epoch: int = 100,
        min_examples_per_bin: int = 512,
    ):
        super().__init__()
        self.enabled = enabled
        self.num_bins = num_bins
        self.epochs_per_bin = epochs_per_bin
        self.overlap = overlap
        # Validate overlap: keep it in a sensible range (0% - 50%). Default is 20%.
        try:
            if not (0.0 <= float(self.overlap) <= 0.5):
                LOGGER.warning(
                    "`overlap` should be between 0.0 and 0.5 (fractions). Clamping to valid range."
                )
                self.overlap = max(0.0, min(float(self.overlap), 0.5))
        except Exception:
            # If overlap isn't a number, fall back to default 0.2
            LOGGER.warning("Invalid `overlap` value; falling back to 0.2 (20%)")
            self.overlap = 0.2
        self.min_train_len = min_train_len
        self.max_train_len = max_train_len
        # If True, compute bins using dataset length quantiles (each bin will
        # contain roughly equal numbers of examples). Otherwise, use uniform
        # length ranges between min/max.
        self.use_quantiles = use_quantiles
        # If True, compute percentiles using an exact full-dataset scan so
        # each bin contains exact counts (subject to duplicates). This will
        # scan the entire dataset and can be slower for very large datasets.
        self.exact_percentiles = exact_percentiles
        # How many examples to sample when estimating dataset length
        self.sample_size = sample_size
        # Sampling strategy for selecting indices: 'linspace' or 'random'
        self.sample_strategy = sample_strategy
        # Minimum number of batches required per epoch (to avoid training instability)
        self.min_batches_per_epoch = min_batches_per_epoch
        # Minimum number of examples required per bin (bins with fewer are skipped)
        self.min_examples_per_bin = min_examples_per_bin

        self.current_bin = 0
        self.epochs_in_current_bin = 0
        self.bin_boundaries = None
        self.tokenizer = None  # Will be set during setup
        self.original_train_dataloader = None  # Store original dataloader
        self.filtered_dataloader = None  # Current filtered dataloader

        if not enabled:
            LOGGER.info("Curriculum learning is disabled")

    def setup(
        self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str
    ) -> None:
        """Setup the curriculum learning boundaries."""
        if not self.enabled or stage != "fit":
            return

        # Store tokenizer for later use
        self.tokenizer = pl_module.tokenizer

        # Store reference to the Lightning module to modify its dataloader
        self.pl_module = pl_module

        # Store the original train_dataloader method if not already stored
        if self.original_train_dataloader is None:
            self.original_train_dataloader = pl_module.train_dataloader

        # Get length range from config if not provided
        if self.min_train_len is None or self.max_train_len is None:
            config = pl_module.config

            # Try to get min_train_len and max_train_len from config
            if hasattr(config.data.properties, "min_train_len") and hasattr(
                config.data.properties, "max_train_len"
            ):
                self.min_train_len = config.data.properties.min_train_len
                self.max_train_len = config.data.properties.max_train_len
            else:
                # For tasks like BFVP and arithmetic that don't have explicit length ranges,
                # we need to infer them from the dataset or use reasonable defaults
                LOGGER.error(
                    "Curriculum learning requires min_train_len and max_train_len to be set "
                    "in the data config (data.properties.min_train_len and data.properties.max_train_len). "
                    "For tasks like BFVP and arithmetic, these should be added to the config file "
                    "to specify the range of sequence lengths to use for curriculum learning."
                )
                raise ValueError(
                    "min_train_len and max_train_len must be set in config.data.properties "
                    "for curriculum learning to work."
                )

        # Calculate bin boundaries with overlap
        self._compute_bin_boundaries()

        # Sample the dataset to see the actual sequence length distribution.
        # If quantile-based bins are requested, compute bins from the sampled
        # lengths. Otherwise, if the observed min/max differ from config,
        # update min/max and recompute uniform bins to avoid empty bins.
        observed = self._observe_dataset_length_range(
            trainer, max_samples=self.sample_size, full_scan=self.exact_percentiles
        )
        if observed is not None:
            # _observe_dataset_length_range now returns (obs_min, obs_max, samples)
            obs_min, obs_max, samples = observed
            if obs_min is not None and obs_max is not None:
                # Update configured range to observed if changed
                if obs_min != self.min_train_len or obs_max != self.max_train_len:
                    LOGGER.info(
                        f"Observed dataset length range: [{obs_min}, {obs_max}]. "
                        "Adjusting curriculum range to observed values and recomputing bins."
                    )
                    self.min_train_len = obs_min
                    self.max_train_len = obs_max

                # If quantile bins requested and we have samples, compute bins
                if self.use_quantiles and samples is not None and len(samples) > 0:
                    bin_boundaries = []
                    if self.exact_percentiles:
                        # Compute exact-count percentile splits from sorted samples
                        sorted_samples = np.sort(samples)
                        N = len(sorted_samples)
                        # cut indices split dataset into nearly equal counts
                        cuts = np.linspace(0, N, self.num_bins + 1, dtype=int)
                        for i in range(self.num_bins):
                            start_idx = cuts[i]
                            end_idx = max(cuts[i + 1] - 1, start_idx)
                            bstart = int(
                                max(self.min_train_len, sorted_samples[start_idx])
                            )
                            bend = int(min(self.max_train_len, sorted_samples[end_idx]))

                            # If the bin is degenerate (same start/end), expand by 1
                            if bend <= bstart:
                                bend = min(self.max_train_len, bstart + 1)

                            # Apply overlap as fraction of bin width
                            width = max(1, bend - bstart)
                            overlap_amount = int(width * self.overlap)
                            bstart = max(self.min_train_len, bstart - overlap_amount)
                            bend = min(self.max_train_len, bend + overlap_amount)

                            # Ensure last bin reaches observed max
                            if i == self.num_bins - 1:
                                bend = self.max_train_len

                            bin_boundaries.append((bstart, bend))
                    else:
                        # Fallback: use quantiles computed from samples (approx)
                        qs = np.quantile(
                            samples, np.linspace(0.0, 1.0, self.num_bins + 1)
                        )
                        for i in range(self.num_bins):
                            bstart = int(max(self.min_train_len, np.floor(qs[i])))
                            bend = int(min(self.max_train_len, np.ceil(qs[i + 1])))

                            width = max(1, bend - bstart)
                            overlap_amount = int(width * self.overlap)
                            bstart = max(self.min_train_len, bstart - overlap_amount)
                            bend = min(self.max_train_len, bend + overlap_amount)

                            if i == self.num_bins - 1:
                                bend = self.max_train_len

                            bin_boundaries.append((bstart, bend))

                    self.bin_boundaries = bin_boundaries
                else:
                    # Use the uniform-length bins computed earlier (or recompute)
                    self._compute_bin_boundaries()

        LOGGER.info(f"Curriculum Learning enabled with {self.num_bins} bins")
        LOGGER.info(f"Length range: [{self.min_train_len}, {self.max_train_len}]")
        LOGGER.info(f"Epochs per bin: {self.epochs_per_bin}")
        LOGGER.info(f"Overlap ratio: {self.overlap} ({self.overlap * 100:.0f}%)")
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

    def _observe_dataset_length_range(
        self, trainer: L.Trainer, max_samples: int = 1000, full_scan: bool = False
    ):
        """
        Inspect the training dataset (up to `max_samples` samples) to estimate the
        observed minimum and maximum sequence lengths and collect sampled lengths.

        Returns a tuple (min_len, max_len, samples_array) where samples_array is a
        numpy array of sampled sequence lengths. Returns None if the dataset
        couldn't be inspected.
        """
        # Attempt to get the original dataloader (callable) first, else fall back
        train_dataloader = None
        if callable(self.original_train_dataloader):
            try:
                train_dataloader = self.original_train_dataloader()
            except Exception:
                train_dataloader = getattr(trainer, "train_dataloader", None)
        else:
            train_dataloader = getattr(trainer, "train_dataloader", None)

        if train_dataloader is None:
            return None

        dataset = getattr(train_dataloader, "dataset", None)
        if dataset is None or len(dataset) == 0:
            return None

        # Choose a set of indices to sample. If full_scan True, inspect entire dataset.
        if full_scan:
            sample_count = len(dataset)
            sample_indices = np.arange(len(dataset), dtype=int)
        else:
            sample_count = min(len(dataset), max_samples)
            if sample_count >= len(dataset):
                sample_indices = np.arange(len(dataset), dtype=int)
            else:
                if self.sample_strategy == "random":
                    rng = np.random.default_rng()
                    sample_indices = rng.choice(
                        len(dataset), size=sample_count, replace=False
                    )
                else:
                    # default: evenly spaced indices
                    sample_indices = np.linspace(
                        0, len(dataset) - 1, sample_count, dtype=int
                    )

        observed_min = None
        observed_max = None
        sample_lengths = []

        for idx in sample_indices:
            try:
                example = dataset[int(idx)]
            except Exception:
                continue

            seq_len = None
            try:
                if isinstance(example, dict) and "text" in example:
                    text = example["text"]
                    if "#" in text:
                        input_part = text.split("#")[0].strip()
                        seq_len = len(input_part.split())
                    else:
                        seq_len = len(text.strip().split())
                elif isinstance(example, dict) and "attention_mask" in example:
                    seq_len = int(example["attention_mask"].sum().item() - 2)
                elif (
                    isinstance(example, dict)
                    and "input_ids" in example
                    and self.tokenizer is not None
                ):
                    input_ids = example["input_ids"]
                    seq_len = int(
                        (input_ids != self.tokenizer.pad_token_id).sum().item() - 2
                    )
            except Exception:
                # If any example is malformed, skip it
                seq_len = None

            if seq_len is None:
                continue

            sample_lengths.append(seq_len)

            if observed_min is None or seq_len < observed_min:
                observed_min = seq_len
            if observed_max is None or seq_len > observed_max:
                observed_max = seq_len

        if observed_min is None or observed_max is None:
            return None

        import numpy as _np

        return (observed_min, observed_max, _np.array(sample_lengths, dtype=int))

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """
        Check if we should move to the next bin at the start of each epoch.

        Note: This only affects TRAINING data. Validation and test sets always use
        the full dataset without any length filtering.
        """
        if not self.enabled:
            return

        # Check if we should advance to next bin
        bin_changed = False
        if (
            self.epochs_in_current_bin >= self.epochs_per_bin
            and self.current_bin < self.num_bins - 1
        ):
            old_bin = self.current_bin
            self.current_bin += 1
            self.epochs_in_current_bin = 0
            bin_changed = True

            old_min, old_max = self.bin_boundaries[old_bin]
            new_min, new_max = self.bin_boundaries[self.current_bin]

            LOGGER.info("")
            LOGGER.info("=" * 80)
            LOGGER.info(
                f"📚 CURRICULUM ADVANCEMENT: Bin {old_bin + 1} → Bin {self.current_bin + 1}"
            )
            LOGGER.info(f"   Previous length range: [{old_min}, {old_max}]")
            LOGGER.info(f"   New length range:      [{new_min}, {new_max}]")
            LOGGER.info(f"   Progress: {self.current_bin + 1}/{self.num_bins} bins")

            # Special message for entering the final bin
            if self.current_bin == self.num_bins - 1:
                LOGGER.info("   Note: Final bin will train until max_steps is reached")

            LOGGER.info("=" * 80)
            LOGGER.info("")

        # Check if we've completed all curriculum bins
        if (
            self.epochs_in_current_bin >= self.epochs_per_bin
            and self.current_bin == self.num_bins - 1
        ):
            # Switch to full dataset after completing all curriculum bins
            LOGGER.info("")
            LOGGER.info("=" * 80)
            LOGGER.info("🎓 CURRICULUM COMPLETED: Switching to FULL dataset")
            LOGGER.info(f"   Trained through all {self.num_bins} bins")
            LOGGER.info("   Now training on ALL lengths until max_steps or early stopping")
            LOGGER.info("=" * 80)
            LOGGER.info("")

            # Restore original dataloader (no filtering)
            if callable(self.original_train_dataloader):
                self.pl_module.train_dataloader = self.original_train_dataloader
            self.epochs_in_current_bin += 1
            return  # Skip filtering - use full dataset from now on

        # Get current bin boundaries
        min_len, max_len = self.bin_boundaries[self.current_bin]

        if not bin_changed:
            # For the last bin, show progress toward completing curriculum
            if self.current_bin == self.num_bins - 1:
                LOGGER.info(
                    f"Epoch {trainer.current_epoch}: Training on length range [{min_len}, {max_len}] "
                    f"(bin {self.current_bin + 1}/{self.num_bins}, epoch {self.epochs_in_current_bin + 1}/{self.epochs_per_bin} - "
                    f"then switching to full dataset)"
                )
            else:
                LOGGER.info(
                    f"Epoch {trainer.current_epoch}: Training on length range [{min_len}, {max_len}] "
                    f"(bin {self.current_bin + 1}/{self.num_bins}, epoch {self.epochs_in_current_bin + 1}/{self.epochs_per_bin})"
                )

        # Apply length filter to dataloader
        self._apply_length_filter(trainer, min_len, max_len)

        self.epochs_in_current_bin += 1

    def _apply_length_filter(
        self, trainer: L.Trainer, min_len: int, max_len: int
    ) -> None:
        """
        Apply length filtering to the training dataloader.

        This creates a filtered view of the dataset that only includes examples
        within the specified length range.

        Note: Length is computed from the raw text (without special tokens like BOS/EOS)
        to match the length ranges specified in the config files.
        """
        train_dataloader = trainer.train_dataloader
        if train_dataloader is None:
            return

        # Get the underlying dataset
        dataset = train_dataloader.dataset

        # Create indices for examples within the length range
        filtered_indices = []
        length_samples = []  # Collect samples to show distribution

        # Debug: log what keys are available in the first example
        if len(dataset) > 0:
            first_example = dataset[0]
            LOGGER.info(f"Dataset example keys: {list(first_example.keys())}")
            if "text" in first_example:
                sample_text = first_example["text"]
                full_len = len(sample_text.strip().split())

                # Also compute input length (before '#')
                if "#" in sample_text:
                    input_part = sample_text.split("#")[0].strip()
                    input_len = len(input_part.split())
                    LOGGER.info(
                        f"Sample text: '{sample_text[:100]}...' "
                        f"(full length: {full_len}, input length: {input_len})"
                    )
                else:
                    LOGGER.info(
                        f"Sample text: '{sample_text[:100]}...' (length: {full_len})"
                    )

        for idx in range(len(dataset)):
            example = dataset[idx]

            # Calculate sequence length from the raw text (without special tokens)
            # This matches the length ranges in config files which are based on raw text
            if "text" in example:
                text = example["text"]

                # For formal language tasks, the length should be based on the INPUT part
                # (before the '#' separator), not the full sequence with traces
                if "#" in text:
                    # Split on '#' and use only the input part
                    input_part = text.split("#")[0].strip()
                    raw_tokens = input_part.split()
                else:
                    # No separator, use full text
                    raw_tokens = text.strip().split()

                seq_len = len(raw_tokens)
            elif "attention_mask" in example:
                # Fallback: count non-padding tokens (includes BOS/EOS if present)
                # Subtract 2 to approximate raw length (assuming BOS + EOS)
                seq_len = example["attention_mask"].sum().item() - 2
            elif "input_ids" in example:
                # Fallback: count tokens that are not padding, minus special tokens
                input_ids = example["input_ids"]
                seq_len = (input_ids != self.tokenizer.pad_token_id).sum().item() - 2
            else:
                continue

            # Collect length samples (every 10000th example) for distribution analysis
            if idx % 10000 == 0:
                length_samples.append(seq_len)

            # Include example if within current bin range
            if min_len <= seq_len <= max_len:
                filtered_indices.append(idx)

        # Show length distribution from samples
        if length_samples:
            import numpy as np

            length_array = np.array(length_samples)
            LOGGER.info(
                f"Length distribution (sampled): min={length_array.min()}, "
                f"max={length_array.max()}, mean={length_array.mean():.1f}, "
                f"median={np.median(length_array):.1f}"
            )

        LOGGER.info(
            f"Filtered dataset: {len(filtered_indices)}/{len(dataset)} examples "
            f"in length range [{min_len}, {max_len}]"
        )

        # Check if bin has minimum required examples
        if len(filtered_indices) < self.min_examples_per_bin:
            LOGGER.warning("")
            LOGGER.warning("=" * 80)
            LOGGER.warning(f"⚠️  INSUFFICIENT EXAMPLES FOR CURRICULUM BIN - SKIPPING")
            LOGGER.warning(f"   Length range [{min_len}, {max_len}] has only {len(filtered_indices)} examples")
            LOGGER.warning(f"   Minimum required: {self.min_examples_per_bin} examples")
            LOGGER.warning(f"   ")
            LOGGER.warning(f"   ACTION: Skipping this bin and advancing to the next one")
            LOGGER.warning("=" * 80)
            LOGGER.warning("")

            # Skip this bin by advancing immediately
            if self.current_bin < self.num_bins - 1:
                self.current_bin += 1
                self.epochs_in_current_bin = 0
                LOGGER.info(f"Advanced to bin {self.current_bin + 1}/{self.num_bins}")
                # Recursively apply filter with new bin
                min_len, max_len = self.bin_boundaries[self.current_bin]
                self._apply_length_filter(trainer, min_len, max_len)
            else:
                # Already at last bin, switch to full dataset
                LOGGER.info("No more bins available, switching to full dataset")
                if callable(self.original_train_dataloader):
                    self.pl_module.train_dataloader = self.original_train_dataloader
            return

        # Check if we have enough examples for stable training (batches per epoch)
        if len(filtered_indices) > 0:
            batch_size = train_dataloader.batch_size
            estimated_batches = len(filtered_indices) // batch_size

            if estimated_batches < self.min_batches_per_epoch:
                LOGGER.warning("")
                LOGGER.warning("=" * 80)
                LOGGER.warning(f"⚠️  INSUFFICIENT DATA FOR CURRICULUM BIN")
                LOGGER.warning(f"   Length range [{min_len}, {max_len}] has only {len(filtered_indices)} examples")
                LOGGER.warning(f"   This gives ~{estimated_batches} batches (batch_size={batch_size})")
                LOGGER.warning(f"   Minimum required: {self.min_batches_per_epoch} batches")
                LOGGER.warning(f"   ")
                LOGGER.warning(f"   SOLUTION: Expanding range to gather more examples...")

                # Expand the range until we have enough examples
                expansion_step = max(1, (max_len - min_len) // 4)  # Expand by 25% increments
                expanded_min = min_len
                expanded_max = max_len

                while estimated_batches < self.min_batches_per_epoch:
                    # Try expanding down first
                    if expanded_min > self.min_train_len:
                        expanded_min = max(self.min_train_len, expanded_min - expansion_step)

                    # Then expanding up
                    if estimated_batches < self.min_batches_per_epoch and expanded_max < self.max_train_len:
                        expanded_max = min(self.max_train_len, expanded_max + expansion_step)

                    # Recount with expanded range
                    filtered_indices = []
                    for idx in range(len(dataset)):
                        example = dataset[idx]
                        if "text" in example:
                            text = example["text"]
                            if "#" in text:
                                input_part = text.split("#")[0].strip()
                                seq_len = len(input_part.split())
                            else:
                                seq_len = len(text.strip().split())
                        elif "attention_mask" in example:
                            seq_len = example["attention_mask"].sum().item() - 2
                        elif "input_ids" in example:
                            input_ids = example["input_ids"]
                            seq_len = (input_ids != self.tokenizer.pad_token_id).sum().item() - 2
                        else:
                            continue

                        if expanded_min <= seq_len <= expanded_max:
                            filtered_indices.append(idx)

                    estimated_batches = len(filtered_indices) // batch_size

                    # Safety: if we've expanded to full range and still don't have enough, break
                    if expanded_min <= self.min_train_len and expanded_max >= self.max_train_len:
                        break

                LOGGER.warning(f"   Expanded to range [{expanded_min}, {expanded_max}]")
                LOGGER.warning(f"   New dataset size: {len(filtered_indices)} examples (~{estimated_batches} batches)")
                LOGGER.warning("=" * 80)
                LOGGER.warning("")

                # Update the range for logging
                min_len = expanded_min
                max_len = expanded_max

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
            # Attach tokenizer to new dataloader for compatibility
            new_dataloader.tokenizer = self.tokenizer

            # Store the filtered dataloader
            self.filtered_dataloader = new_dataloader

            # Replace dataloader by overriding the train_dataloader method on the Lightning module
            # This is the most reliable way that works across Lightning versions
            callback_self = self  # Capture in closure

            def curriculum_train_dataloader():
                """Return the curriculum-filtered dataloader."""
                if callback_self.filtered_dataloader is not None:
                    return callback_self.filtered_dataloader
                # Fallback to original
                return (
                    callback_self.original_train_dataloader()
                    if callable(callback_self.original_train_dataloader)
                    else callback_self.original_train_dataloader
                )

            # Replace the method
            self.pl_module.train_dataloader = curriculum_train_dataloader
        else:
            LOGGER.warning(
                f"No examples found in length range [{min_len}, {max_len}]. "
                "Keeping full dataset."
            )

    def on_validation_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Log that validation uses full dataset."""
        _ = pl_module  # Unused but required by Lightning API
        if self.enabled and trainer.current_epoch == 0:
            # Only log once at the first validation to avoid spam
            LOGGER.info(
                "Validation/Test: Using FULL dataset (curriculum filtering only applies to training)"
            )

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Log curriculum learning completion."""
        _ = trainer, pl_module  # Unused but required by Lightning API
        if self.enabled:
            LOGGER.info("")
            LOGGER.info("=" * 80)
            LOGGER.info("✅ Curriculum learning completed!")
            LOGGER.info(f"   Trained through all {self.num_bins} bins")
            LOGGER.info(f"   Final length range: {self.bin_boundaries[-1]}")
            LOGGER.info("=" * 80)
            LOGGER.info("")

    def state_dict(self):
        """Save callback state for checkpointing."""
        return {
            "current_bin": self.current_bin,
            "epochs_in_current_bin": self.epochs_in_current_bin,
            "bin_boundaries": self.bin_boundaries,
        }

    def load_state_dict(self, state_dict):
        """Load callback state from checkpoint."""
        self.current_bin = state_dict["current_bin"]
        self.epochs_in_current_bin = state_dict["epochs_in_current_bin"]
        self.bin_boundaries = state_dict["bin_boundaries"]
        LOGGER.info(
            f"Resumed curriculum learning at bin {self.current_bin + 1}/{self.num_bins}, "
            f"epoch {self.epochs_in_current_bin + 1}/{self.epochs_per_bin}"
        )
