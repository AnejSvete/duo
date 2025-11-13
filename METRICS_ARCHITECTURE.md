# Metrics Computation Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        PyTorch Lightning Trainer                  │
│                                                                   │
│  fit() ─→ Validation Loop ──→ on_validation_epoch_end()          │
│           test() ─→ Test Loop ─→ on_test_epoch_end()             │
└─────────────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────────┐
│                  TrainerBase (extends L.LightningModule)         │
│                                                                   │
│  validation_step(batch)          test_step(batch)               │
│  ├─ _loss()                      ├─ _loss()                      │
│  ├─ metrics.update_valid()       ├─ metrics.update_valid()       │
│  ├─ _extract_prompts_targets()   ├─ _extract_prompts_targets()   │
│  ├─ generate_conditioned()       ├─ generate_conditioned()       │
│  ├─ _compute_accuracy()          ├─ _compute_accuracy()          │
│  └─ self.log() → W&B             └─ self.log() → W&B             │
│                                                                   │
│  on_validation_epoch_end()       on_test_epoch_end()            │
│  ├─ metrics.valid_nlls.compute() ├─ metrics.valid_nlls.compute() │
│  ├─ self.log() → W&B             ├─ self.log() → W&B             │
│  └─ Save validation_metrics.json └─ Save test_metrics.json       │
└─────────────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Metrics Aggregation (metrics.py)              │
│                                                                   │
│  self.valid_nlls (MetricCollection)                              │
│  ├─ 'val/nll' (NLL class)      ─ Weighted mean aggregation       │
│  ├─ 'val/bpd' (BPD class)      ─ nll / log(2)                    │
│  └─ 'val/ppl' (Perplexity)     ─ exp(nll / num_tokens)           │
│                                                                   │
│  update_valid(nll, prior_loss, num_tokens)                       │
│  └─ Updates running mean: mean_value += nll.sum()               │
│                            weight += num_tokens.sum()            │
│                                                                   │
│  compute() → Final aggregated scalar value                       │
└─────────────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Output & Persistence Layer                 │
│                                                                   │
│  Weights & Biases                   File System                   │
│  ├─ val/nll ─────────────────────→ Dashboard              │
│  ├─ val/bpd                                               │
│  ├─ val/ppl                        validation_metrics.json│
│  ├─ val/*_acc_*                    └─ Cumulative JSON     │
│  ├─ conditioned_generation@* table │  (append per epoch)   │
│  │                                                         │
│  └─ test/*, test_* tables          test_metrics.json       │
│                                    └─ Single JSON          │
│                                       (overwrite)          │
└─────────────────────────────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Analysis & Visualization (analyze_metrics.py) │
│                                                                   │
│  load_validation_metrics() ─→ pandas DataFrame                   │
│  load_test_metrics() ──────→ dictionary                          │
│  plot_validation_metrics() → validation_metrics_plot.png         │
│  create_validation_summary_table() → validation_summary.csv      │
│  plot_test_metrics_comparison() → test_metrics_plot.png          │
│  create_publication_table() → test_results_table.tex             │
└─────────────────────────────────────────────────────────────────┘
```

## Per-Batch Metrics Computation Flow

```
Input Batch Dictionary
{
  'input_ids': (batch_size, seq_len),
  'attention_mask': (batch_size, seq_len),  # 1=real, 0=padding
  'do_not_mask': (batch_size, seq_len)      # True=prompt, False=target
}
    ↓
┌──────────────────────────────────────┐
│  STEP 1: Loss Computation            │
├──────────────────────────────────────┤
│ trainer_base._loss()                 │
│ ├─ _process_model_input()            │
│ │  └─ Algorithm-specific input prep  │
│ └─ nll() [algorithm-specific]        │
│    ├─ Forward pass: model(input)     │
│    └─ Compute per-token NLL          │
│                                      │
│ Returns: Loss(loss, nlls, prior_loss,│
│                num_tokens)           │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  STEP 2: Metrics Update              │
├──────────────────────────────────────┤
│ metrics.update_valid(nlls, prior,    │
│                     num_tokens)      │
│                                      │
│ Running aggregation in:              │
│ metrics.valid_nlls['nll'].mean_value │
│ metrics.valid_nlls['nll'].weight     │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  STEP 3: Accuracy Computation        │
├──────────────────────────────────────┤
│ _extract_prompts_and_targets(        │
│   input_ids, do_not_mask)            │
│                                      │
│ prompts = input_ids.clone()          │
│ prompts[~do_not_mask & not_pad] =    │
│   tokenizer.mask_token_id            │
│                                      │
│ targets = input_ids.clone()          │
│ targets[do_not_mask] =               │
│   tokenizer.pad_token_id             │
│                                      │
│ Returns: (prompts, targets)          │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  STEP 4: Generation                  │
├──────────────────────────────────────┤
│ generate_conditioned(                │
│   prompts, targets,                  │
│   mode=gen_mode, top_k=top_k)        │
│                                      │
│ AR (algo.py:9-58)                   │
│ └─ Token-by-token greedy decoding    │
│                                      │
│ LT (algo.py:125-150)                │
│ └─ Single-pass masking               │
│                                      │
│ MDLM (diffusion.py + algo.py)       │
│ ├─ random: Random unmasking          │
│ ├─ top_k: Confidence-based           │
│ ├─ one_level: Hierarchical (|)       │
│ ├─ all_at_once: One-step             │
│ └─ one_at_a_time: Left-to-right      │
│                                      │
│ Returns: generated (batch_size, seq) │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  STEP 5: Accuracy Metrics            │
├──────────────────────────────────────┤
│ _compute_accuracy(generated, targets)│
│                                      │
│ target_mask = targets != pad_token   │
│                                      │
│ 1. acc_exact:                        │
│    % sequences where all target      │
│    tokens match exactly              │
│    = (generated == targets)          │
│      .all(dim=1).float().mean()      │
│                                      │
│ 2. acc_token:                        │
│    Token-level accuracy              │
│    = sum(gen==tgt & tgt_mask) /     │
│      sum(tgt_mask)                   │
│                                      │
│ 3. correct_prediction:               │
│    % sequences with correct final    │
│    answer token                      │
│    = (gen[last_idx]==tgt[last_idx])  │
│      .mean()                         │
│                                      │
│ Returns: (acc_exact, acc_token,      │
│           correct_prediction)        │
└──────────────────────────────────────┘
    ↓
┌──────────────────────────────────────┐
│  STEP 6: Per-Batch Logging           │
├──────────────────────────────────────┤
│ self.log(                            │
│   f'val/{gen_mode}_acc_exact',       │
│   acc_exact,                         │
│   on_step=False,                     │
│   on_epoch=True,                     │
│   sync_dist=True)                    │
│                                      │
│ Lightning accumulates these per      │
│ epoch for final aggregation          │
│                                      │
│ W&B logger receives metric for       │
│ each generation mode                 │
└──────────────────────────────────────┘
```

## Epoch-End Aggregation Flow

```
[All Validation Batches Processed]
    ↓
┌──────────────────────────────────────┐
│  on_validation_epoch_end()           │
└──────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────┐
│  STEP 1: Compute Final Loss Metrics            │
├────────────────────────────────────────────────┤
│ for k, v in metrics.valid_nlls.items():        │
│   final_value = v.compute()                    │
│                                                │
│ Computation details:                           │
│ - NLL: accumulated_mean_value /                │
│        accumulated_weight                      │
│ - BPD: nll / log(2)                            │
│ - PPL: exp(nll / num_tokens)                   │
│                                                │
│ Synchronized across all GPUs with sync_dist=T │
└────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────┐
│  STEP 2: Log to W&B                            │
├────────────────────────────────────────────────┤
│ self.log('val/nll', nll_value, on_epoch=True) │
│ self.log('val/bpd', bpd_value, on_epoch=True) │
│ self.log('val/ppl', ppl_value, on_epoch=True) │
│                                                │
│ W&B receives scalar metrics per epoch          │
└────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────┐
│  STEP 3: Prepare JSON Save                     │
├────────────────────────────────────────────────┤
│ current_metrics = {                            │
│   'val/nll': nll_value.item(),                 │
│   'val/bpd': bpd_value.item(),                 │
│   'val/ppl': ppl_value.item(),                 │
│   'epoch': current_epoch,                      │
│   'global_step': global_step                   │
│ }                                              │
│                                                │
│ Also add callback metrics from                 │
│ self.trainer.callback_metrics:                 │
│ ├─ val/*_acc_exact (per gen_mode)              │
│ ├─ val/*_acc_token (per gen_mode)              │
│ └─ val/*_correct_prediction (per gen_mode)     │
└────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────┐
│  STEP 4: Save to validation_metrics.json       │
├────────────────────────────────────────────────┤
│ if validation_metrics.json exists:             │
│   - Load existing JSON (list of dicts)         │
│ else:                                          │
│   - Create empty list                          │
│                                                │
│ Append current_metrics to list                 │
│                                                │
│ Overwrite JSON with updated list               │
│                                                │
│ Result: Cumulative history of all validation   │
│ epochs in a single array                       │
└────────────────────────────────────────────────┘
    ↓
┌────────────────────────────────────────────────┐
│  STEP 5: Resume Training                       │
├────────────────────────────────────────────────┤
│ self._train_mode()                             │
│ - Set backbone to train()                      │
│ - Set noise to train()                         │
│ - Continue gradient accumulation               │
└────────────────────────────────────────────────┘
```

## Test Epoch Flow (Simplified)

```
[Test Run Initiated]
    ↓
on_test_epoch_start()
├─ metrics.reset()
├─ _eval_mode()
└─ Assert clean state
    ↓
test_step() [per batch] ─ Same as validation_step
    ├─ _loss() → nlls
    ├─ metrics.update_valid(nlls)
    ├─ _extract_prompts_and_targets()
    ├─ generate_conditioned()
    ├─ _compute_accuracy()
    └─ self.log() → W&B
    ↓
on_test_epoch_end()
├─ metrics.valid_nlls.compute() → final values
├─ self.log('test/*', values) → W&B
└─ Save to test_metrics.json (OVERWRITES)
    ↓
_train_mode() ─ Resume training (if resuming)
```

## Data Availability at Each Step

```
validation_step(batch, batch_idx)
│
├─ batch['input_ids']        : (batch, seq_len) - token IDs
├─ batch['attention_mask']   : (batch, seq_len) - 0/1 mask
├─ batch['do_not_mask']      : (batch, seq_len) - prompt mask
│
├─ self.config.model.length  : Fixed seq length (e.g., 256)
├─ self.config.eval.top_k    : For generation
├─ self.config.algo.name     : 'ar' / 'lt' / 'mdlm' / etc.
│
├─ self.tokenizer.pad_token_id
├─ self.tokenizer.mask_token_id
├─ self.tokenizer.vocab_size
│
├─ self.metrics.valid_nlls   : Current running metrics
│
├─ self.global_step
├─ self.current_epoch
│
└─ self.trainer.global_rank  : For distributed training
   self.trainer.num_gpus
   self.config.checkpointing.save_dir
```

## Metrics File Locations

```
${checkpointing.save_dir}/
├─ validation_metrics.json     # Cumulative array [epoch0, epoch1, ...]
├─ test_metrics.json           # Single object (final test results)
├─ checkpoints/
│  └─ last.ckpt              # Latest checkpoint
└─ outputs/
   └─ ... (Hydra output files)
```

## Generation Mode Dispatch

```
validation_step → gen_modes determination
    ↓
IF algo.name == 'mdlm':
    gen_modes = ['random', 'top_k', 'one_level', 
                 'all_at_once', 'one_at_a_time']
    └─ Each mode generates separately
       └─ Each logs acc_exact, acc_token, correct_prediction
ELSE:  # AR or LT
    gen_modes = ['default']
    └─ Single generation
       └─ Logs val/acc_token (summary metric only)
    
Special case for MDLM:
    top_k results mapped to 'default_*' metrics
    └─ For fair comparison with AR/LT
```

## W&B Metric Grouping

```
trainer/
├─ loss           : Per step
└─ lr             : Learning rate per step

val/
├─ nll            : Epoch aggregated
├─ bpd            : Epoch aggregated
├─ ppl            : Epoch aggregated
├─ default_acc_exact
├─ default_acc_token
├─ default_correct_prediction
├─ random_acc_exact          [MDLM only]
├─ random_acc_token
├─ random_correct_prediction
├─ top_k_acc_exact
├─ top_k_acc_token
├─ ...                       [Similar for one_level, all_at_once, one_at_a_time]
├─ acc_token                 [Summary metric - mapped from best mode]
└─ conditioned_generation@global_step{N}  : Sample table

test/
├─ nll            : Final test
├─ bpd
├─ ppl
├─ default_*                 [Same structure as val/]
├─ random_*                  [MDLM only]
├─ top_k_*
├─ one_level_*
├─ all_at_once_*
├─ one_at_a_time_*
├─ acc_token
└─ test_conditioned_generation@global_step{N}
```

