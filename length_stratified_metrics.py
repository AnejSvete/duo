"""Length-stratified metrics tracking for model evaluation.

This module provides utilities for tracking and aggregating model performance
metrics stratified by sequence length during validation and testing.
"""

import typing
from collections import defaultdict

import numpy as np
import torch


class LengthStratifiedMetrics:
    """Tracks metrics stratified by sequence length bins.

    Computes percentile-based bins for sequence lengths and maintains
    separate metric accumulators for each bin, plus overall metrics.
    """

    def __init__(self, num_bins: int = 10, percentile_based: bool = True):
        """Initialize length-stratified metrics tracker.

        Args:
            num_bins: Number of bins to split lengths into (default: 10 for deciles)
            percentile_based: If True, use percentile-based binning, otherwise uniform
        """
        self.num_bins = num_bins
        self.percentile_based = percentile_based
        self.reset()

    def reset(self):
        """Reset all metrics and length statistics."""
        # Store all lengths seen to compute percentiles
        self.all_lengths = []

        # Metrics per bin: {bin_idx: {metric_name: [values]}}
        self.bin_metrics = defaultdict(lambda: defaultdict(list))

        # Overall metrics (not binned)
        self.overall_metrics = defaultdict(list)

        # Bin edges (computed after seeing all data)
        self.bin_edges = None
        self.bin_labels = None

    def update(
        self,
        lengths: torch.Tensor,
        metrics: typing.Dict[str, float],
        weights: typing.Optional[torch.Tensor] = None,
    ):
        """Update metrics for a batch of sequences.

        Args:
            lengths: Tensor of shape (batch_size,) with sequence lengths
            metrics: Dict mapping metric names to batch-level values
            weights: Optional weights for each sequence (default: uniform)
        """
        # Convert lengths to list and store
        lengths_list = lengths.cpu().tolist()
        self.all_lengths.extend(lengths_list)

        # Store overall metrics (weighted if provided)
        for metric_name, metric_value in metrics.items():
            if weights is not None:
                # Store (value, weight) pairs for weighted averaging later
                for i, length in enumerate(lengths_list):
                    weight = weights[i].item() if torch.is_tensor(weights) else 1.0
                    self.overall_metrics[metric_name].append((metric_value, weight))
            else:
                self.overall_metrics[metric_name].append(metric_value)

    def update_per_sample(
        self, lengths: torch.Tensor, per_sample_metrics: typing.Dict[str, torch.Tensor]
    ):
        """Update metrics with per-sample values (more accurate).

        Args:
            lengths: Tensor of shape (batch_size,) with sequence lengths
            per_sample_metrics: Dict mapping metric names to tensors of shape (batch_size,)
        """
        lengths_list = lengths.cpu().tolist()
        self.all_lengths.extend(lengths_list)

        # Store per-sample metrics for both overall and bin-specific
        for metric_name, metric_tensor in per_sample_metrics.items():
            metric_values = metric_tensor.cpu().tolist()
            for length, value in zip(lengths_list, metric_values):
                self.overall_metrics[metric_name].append(value)

    def _compute_bin_edges(self):
        """Compute bin edges from observed lengths."""
        if len(self.all_lengths) == 0:
            return

        lengths_array = np.array(self.all_lengths)

        if self.percentile_based:
            # Compute percentile-based bins
            percentiles = np.linspace(0, 100, self.num_bins + 1)
            self.bin_edges = np.percentile(lengths_array, percentiles)
            # Store the actual percentile values for labels
            self.bin_percentiles = percentiles
            # Ensure unique edges (in case of repeated lengths)
            self.bin_edges = np.unique(self.bin_edges)
        else:
            # Uniform bins
            min_len, max_len = lengths_array.min(), lengths_array.max()
            self.bin_edges = np.linspace(min_len, max_len, self.num_bins + 1)
            self.bin_percentiles = None

        # Create bin labels using percentiles for better cross-dataset comparison
        self.bin_labels = []
        if self.percentile_based:
            # Use percentile ranges in labels (e.g., "p0-25", "p25-50")
            for i in range(len(self.bin_edges) - 1):
                # Find corresponding percentile indices
                p_start = int(
                    self.bin_percentiles[i] if i < len(self.bin_percentiles) else 0
                )
                p_end = int(
                    self.bin_percentiles[i + 1]
                    if i + 1 < len(self.bin_percentiles)
                    else 100
                )
                label = f"p{p_start}-{p_end}"
                self.bin_labels.append(label)
        else:
            # For uniform bins, use actual length ranges
            for i in range(len(self.bin_edges) - 1):
                label = f"len_{int(self.bin_edges[i])}-{int(self.bin_edges[i+1])}"
                self.bin_labels.append(label)

    def _assign_to_bins(self):
        """Assign all collected metrics to appropriate length bins."""
        if self.bin_edges is None:
            self._compute_bin_edges()

        if len(self.bin_edges) < 2:
            # Not enough data to create bins
            return

        # Build a list of metric names to maintain consistent ordering
        metric_names = list(self.overall_metrics.keys())
        if not metric_names:
            return

        # Verify all metrics have the same length
        num_samples = len(self.all_lengths)
        for metric_name in metric_names:
            if len(self.overall_metrics[metric_name]) != num_samples:
                raise ValueError(
                    f"Metric {metric_name} has {len(self.overall_metrics[metric_name])} values "
                    f"but expected {num_samples}"
                )

        # Assign each sample to its appropriate bin
        for i, length in enumerate(self.all_lengths):
            # Find which bin this length belongs to
            bin_idx = np.digitize(length, self.bin_edges[1:-1])
            bin_idx = min(bin_idx, len(self.bin_labels) - 1)  # Clamp to valid range

            # Add all metrics for this sample to the bin
            for metric_name in metric_names:
                metric_value = self.overall_metrics[metric_name][i]
                self.bin_metrics[bin_idx][metric_name].append(metric_value)

    def compute(self) -> typing.Dict[str, typing.Any]:
        """Compute aggregated metrics for all bins and overall.

        Returns:
            Dict with keys:
                - 'overall': Dict of overall metric means
                - 'bins': List of dicts with bin-specific metrics
                - 'bin_edges': Bin edge values
                - 'num_samples': Number of samples in dataset
                - 'length_statistics': Detailed statistics about length distribution
        """
        if len(self.all_lengths) == 0:
            return {
                "overall": {},
                "bins": [],
                "bin_edges": [],
                "num_samples": 0,
                "length_statistics": {},
            }

        # Compute bin assignments
        self._assign_to_bins()

        # Compute overall metrics
        overall = {}
        for metric_name, values in self.overall_metrics.items():
            if len(values) > 0:
                if isinstance(values[0], tuple):
                    # Weighted average
                    total = sum(v * w for v, w in values)
                    weight = sum(w for _, w in values)
                    overall[metric_name] = total / weight if weight > 0 else 0.0
                else:
                    # Simple average
                    overall[metric_name] = float(np.mean(values))

        # Compute length statistics for later reference
        lengths_array = np.array(self.all_lengths)
        length_statistics = {
            "min": float(lengths_array.min()),
            "max": float(lengths_array.max()),
            "mean": float(lengths_array.mean()),
            "std": float(lengths_array.std()),
            "median": float(np.median(lengths_array)),
            "percentiles": {
                f"p{int(p)}": float(np.percentile(lengths_array, p))
                for p in [0, 25, 50, 75, 100]
            },
        }

        # Compute per-bin metrics
        bins_list = []
        for bin_idx in range(len(self.bin_labels)):
            bin_dict = {
                "label": self.bin_labels[bin_idx],
                "min_length": float(self.bin_edges[bin_idx]),
                "max_length": float(self.bin_edges[bin_idx + 1]),
                "num_samples": 0,
            }

            if bin_idx in self.bin_metrics:
                bin_data = self.bin_metrics[bin_idx]
                bin_dict["num_samples"] = len(bin_data[list(bin_data.keys())[0]])

                for metric_name, values in bin_data.items():
                    if len(values) > 0:
                        bin_dict[metric_name] = float(np.mean(values))

            bins_list.append(bin_dict)

        return {
            "overall": overall,
            "bins": bins_list,
            "bin_edges": self.bin_edges.tolist() if self.bin_edges is not None else [],
            "num_samples": len(self.all_lengths),
            "length_statistics": length_statistics,
        }

    def get_wandb_logs(self, prefix: str = "val") -> typing.Dict[str, float]:
        """Get metrics formatted for wandb logging.

        Args:
            prefix: Prefix for metric names (e.g., "val" or "test")

        Returns:
            Dict mapping metric names to values for wandb.log()
        """
        results = self.compute()
        logs = {}

        # Overall metrics (these are already logged elsewhere, but include for completeness)
        for metric_name, value in results["overall"].items():
            logs[f"{prefix}/{metric_name}_overall"] = value

        # Per-bin metrics with percentile-based labels
        for bin_dict in results["bins"]:
            bin_label = bin_dict["label"]
            for metric_name, value in bin_dict.items():
                if metric_name not in [
                    "label",
                    "min_length",
                    "max_length",
                    "num_samples",
                ]:
                    logs[f"{prefix}/num_samples/{metric_name}_{bin_label}"] = value
            # Also log sample count per bin
            logs[f"{prefix}/num_samples/num_samples_{bin_label}"] = bin_dict[
                "num_samples"
            ]

        # Log length statistics for reference
        if "length_statistics" in results and results["length_statistics"]:
            stats = results["length_statistics"]
            logs[f"{prefix}/bin_length/length_min"] = stats["min"]
            logs[f"{prefix}/bin_length/length_max"] = stats["max"]
            logs[f"{prefix}/bin_length/length_mean"] = stats["mean"]
            logs[f"{prefix}/bin_length/length_std"] = stats["std"]
            logs[f"{prefix}/bin_length/length_median"] = stats["median"]
            # Log key percentiles
            for p_name, p_value in stats["percentiles"].items():
                logs[f"{prefix}/bin_length/length_{p_name}"] = p_value

        return logs


class PerGenerationModeMetrics:
    """Manages separate LengthStratifiedMetrics for each generation mode."""

    def __init__(self, num_bins: int = 10):
        """Initialize metrics tracker for multiple generation modes.

        Args:
            num_bins: Number of length bins (default: 10 for deciles)
        """
        self.num_bins = num_bins
        self.metrics_by_mode = {}

    def reset(self):
        """Reset all metrics for all generation modes."""
        self.metrics_by_mode = {}

    def get_or_create_mode_metrics(self, mode: str) -> LengthStratifiedMetrics:
        """Get metrics tracker for a specific generation mode.

        Args:
            mode: Generation mode name (e.g., "default", "top_k", etc.)

        Returns:
            LengthStratifiedMetrics instance for this mode
        """
        if mode not in self.metrics_by_mode:
            self.metrics_by_mode[mode] = LengthStratifiedMetrics(num_bins=self.num_bins)
        return self.metrics_by_mode[mode]

    def update(
        self,
        mode: str,
        lengths: torch.Tensor,
        per_sample_metrics: typing.Dict[str, torch.Tensor],
    ):
        """Update metrics for a specific generation mode.

        Args:
            mode: Generation mode name
            lengths: Tensor of sequence lengths
            per_sample_metrics: Dict of per-sample metric tensors
        """
        metrics = self.get_or_create_mode_metrics(mode)
        metrics.update_per_sample(lengths, per_sample_metrics)

    def compute_all(self) -> typing.Dict[str, typing.Any]:
        """Compute metrics for all generation modes.

        Returns:
            Dict mapping mode names to their computed metrics
        """
        return {
            mode: metrics.compute() for mode, metrics in self.metrics_by_mode.items()
        }

    def get_wandb_logs(self, prefix: str = "val") -> typing.Dict[str, float]:
        """Get all metrics formatted for wandb logging.

        Args:
            prefix: Prefix for metric names (e.g., "val" or "test")

        Returns:
            Dict mapping metric names to values for wandb.log()
        """
        logs = {}
        for mode, metrics in self.metrics_by_mode.items():
            # Use separate W&B panel for each generation method
            mode_logs = metrics.get_wandb_logs(prefix=f"{prefix}_{mode}")
            logs.update(mode_logs)
        return logs
