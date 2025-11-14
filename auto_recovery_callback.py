"""
Automatic checkpoint recovery callback for Lightning.

This callback monitors training for signs of instability (loss explosion, NaN/Inf)
and automatically restores the best checkpoint when detected.
"""
import os
import torch
import lightning as L
from lightning.pytorch.callbacks import Callback


class AutoRecoveryCallback(Callback):
    """
    Automatically recovers from training instability by loading the best checkpoint.

    Monitors:
    - Loss explosions (loss > threshold)
    - NaN/Inf in loss values
    - Gradient explosions (optional)

    When instability is detected, it:
    1. Loads the best checkpoint (based on monitored metric)
    2. Resets optimizer state (optional)
    3. Continues training from the recovered checkpoint

    Args:
        monitor: Metric to use for determining best checkpoint (default: "val/acc_token")
        mode: "min" or "max" for the monitored metric (default: "max")
        loss_threshold: Maximum acceptable loss value (default: 100.0)
        patience: Number of steps to wait before triggering recovery (default: 3)
        reset_optimizer: Whether to reset optimizer state after recovery (default: True)
        checkpoint_dir: Directory containing checkpoints (default: "checkpoints")
        verbose: Print recovery messages (default: True)
    """

    def __init__(
        self,
        monitor="val/acc_token",
        mode="max",
        loss_threshold=100.0,
        patience=3,
        reset_optimizer=True,
        checkpoint_dir="checkpoints",
        verbose=True,
    ):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.loss_threshold = loss_threshold
        self.patience = patience
        self.reset_optimizer = reset_optimizer
        self.checkpoint_dir = checkpoint_dir
        self.verbose = verbose

        # Track consecutive bad steps
        self.bad_steps = 0
        self.recovery_count = 0
        self.last_good_step = 0

        # Track if we're in a recovery phase (to avoid infinite loops)
        self.recovering = False
        self.max_recoveries = 3  # Maximum number of recovery attempts

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Check for instability after each training batch."""

        # Skip if we're already recovering or exceeded max recoveries
        if self.recovering:
            return

        if self.recovery_count >= self.max_recoveries:
            if self.verbose:
                print(f"\n{'='*80}")
                print(f"Maximum recovery attempts ({self.max_recoveries}) reached. Stopping training.")
                print(f"{'='*80}\n")
            trainer.should_stop = True
            return

        # Extract loss from outputs
        if outputs is None or 'loss' not in outputs:
            return

        loss = outputs['loss']

        # Convert to scalar if tensor
        if isinstance(loss, torch.Tensor):
            loss_value = loss.item()
        else:
            loss_value = loss

        # Check for instability
        is_unstable = False
        reason = None

        if torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)):
            is_unstable = True
            reason = "NaN/Inf in loss"
        elif loss_value > self.loss_threshold:
            is_unstable = True
            reason = f"Loss explosion (loss={loss_value:.2f} > threshold={self.loss_threshold})"

        if is_unstable:
            self.bad_steps += 1
            if self.verbose and self.bad_steps == 1:
                print(f"\n{'='*80}")
                print(f"WARNING: Instability detected at step {trainer.global_step}")
                print(f"Reason: {reason}")
                print(f"Waiting {self.patience} steps before recovery...")
                print(f"{'='*80}\n")
        else:
            # Reset bad steps counter if we're back to normal
            if self.bad_steps > 0:
                if self.verbose:
                    print(f"Loss stabilized at step {trainer.global_step}. Resetting counter.")
            self.bad_steps = 0
            self.last_good_step = trainer.global_step

        # Trigger recovery if patience exceeded
        if self.bad_steps >= self.patience:
            self._trigger_recovery(trainer, pl_module, reason)

    def _trigger_recovery(self, trainer, pl_module, reason):
        """Load best checkpoint and reset optimizer state."""

        self.recovering = True
        self.recovery_count += 1

        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TRIGGERING AUTOMATIC RECOVERY (Attempt {self.recovery_count}/{self.max_recoveries})")
            print(f"{'='*80}")
            print(f"Reason: {reason}")
            print(f"Current step: {trainer.global_step}")
            print(f"Last good step: {self.last_good_step}")

        # Find best checkpoint
        best_checkpoint_path = self._find_best_checkpoint(trainer)

        if best_checkpoint_path is None:
            if self.verbose:
                print("ERROR: No checkpoint found for recovery!")
                print(f"{'='*80}\n")
            self.recovering = False
            trainer.should_stop = True
            return

        if self.verbose:
            print(f"Loading checkpoint: {best_checkpoint_path}")

        try:
            # Load checkpoint
            checkpoint = torch.load(best_checkpoint_path, map_location=pl_module.device)

            # Restore model state
            pl_module.load_state_dict(checkpoint['state_dict'])

            # Optionally reset optimizer state
            if self.reset_optimizer:
                if self.verbose:
                    print("Resetting optimizer state...")
                # After loading state_dict, model parameters are new objects
                # We need to recreate the optimizers with the new parameters
                # This is the proper way to reset optimizer state

                # Get new optimizer configuration from the module
                optimizer_config = pl_module.configure_optimizers()

                # Replace trainer's optimizers with fresh ones
                # configure_optimizers() returns ([optimizer], [scheduler_dict])
                if isinstance(optimizer_config, (list, tuple)) and len(optimizer_config) == 2:
                    optimizers, schedulers = optimizer_config
                    trainer.optimizers = optimizers
                    if schedulers:
                        trainer.lr_scheduler_configs = schedulers
                else:
                    # Fallback for other return types
                    trainer.optimizers = [optimizer_config] if not isinstance(optimizer_config, list) else optimizer_config
            else:
                # Restore optimizer state
                if 'optimizer_states' in checkpoint:
                    for opt_idx, optimizer in enumerate(trainer.optimizers):
                        optimizer.load_state_dict(checkpoint['optimizer_states'][opt_idx])

            # Restore global step if needed
            # Note: We don't restore global_step to continue from where we were
            # trainer.global_step = checkpoint.get('global_step', trainer.global_step)

            if self.verbose:
                print(f"Successfully recovered from checkpoint")
                print(f"Checkpoint was from step: {checkpoint.get('global_step', 'unknown')}")
                print(f"Continuing from current step: {trainer.global_step}")
                print(f"{'='*80}\n")

            # Reset counters
            self.bad_steps = 0
            self.recovering = False

        except Exception as e:
            if self.verbose:
                print(f"ERROR loading checkpoint: {e}")
                print(f"{'='*80}\n")
            self.recovering = False
            trainer.should_stop = True

    def _find_best_checkpoint(self, trainer):
        """Find the best checkpoint based on the monitored metric."""

        # Get checkpoint directory
        if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback is not None:
            # Try to get from checkpoint callback
            if hasattr(trainer.checkpoint_callback, 'best_model_path'):
                best_path = trainer.checkpoint_callback.best_model_path
                if os.path.exists(best_path):
                    return best_path

        # Fallback: look for best.ckpt in the checkpoint directory
        checkpoint_dir = os.path.join(trainer.default_root_dir, self.checkpoint_dir)

        # Try best.ckpt first
        best_ckpt = os.path.join(checkpoint_dir, "best.ckpt")
        if os.path.exists(best_ckpt):
            return best_ckpt

        # Fallback: try last.ckpt
        last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            if self.verbose:
                print("Warning: best.ckpt not found, using last.ckpt")
            return last_ckpt

        # Last resort: find any checkpoint
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if checkpoints:
                # Sort by modification time, use most recent
                checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)), reverse=True)
                if self.verbose:
                    print(f"Warning: Using most recent checkpoint: {checkpoints[0]}")
                return os.path.join(checkpoint_dir, checkpoints[0])

        return None

    def on_train_end(self, trainer, pl_module):
        """Print summary at end of training."""
        if self.verbose and self.recovery_count > 0:
            print(f"\n{'='*80}")
            print(f"TRAINING SUMMARY")
            print(f"{'='*80}")
            print(f"Total recovery attempts: {self.recovery_count}")
            print(f"Final step: {trainer.global_step}")
            print(f"{'='*80}\n")
