# Metrics Documentation Index

## Overview

This directory contains comprehensive documentation on validation and testing metrics computation, logging, and aggregation in the duo research codebase. These documents provide everything you need to understand how the system tracks model performance and persists metrics.

---

## Documentation Files

### 1. ANALYSIS_SUMMARY.md (245 lines)
**Quick overview and key findings**

Start here if you:
- Need a 5-minute understanding of the metrics system
- Want to know what's currently being logged
- Need to identify gaps or missing features
- Are looking for a roadmap through the other documents

**Contains**:
- Key findings about metrics computation
- Critical code locations (table)
- Data flow summary
- Configuration parameters
- Known gaps/missing features
- Navigation guide

---

### 2. METRICS_QUICK_REFERENCE.md (210 lines)
**Fast lookup guide for developers**

Use this for:
- Quick code lookups during development
- Finding function locations
- Understanding metric definitions
- Accessing metrics programmatically
- Configuration values reference

**Contains**:
- Key files list
- Critical function snippets
- Three accuracy metric definitions
- Metrics saved to W&B
- JSON file structures with examples
- Batch information available
- Generation modes by algorithm
- Configuration values
- Data flow summary
- Code examples

---

### 3. METRICS_ARCHITECTURE.md (392 lines)
**Visual system architecture and data flows**

Read this for:
- Understanding the big picture
- Seeing how components interact
- Following data through the system
- Understanding process flows
- Identifying where information is available

**Contains**:
- System overview diagram
- Per-batch metrics computation flow
- Epoch-end aggregation flow
- Test epoch flow
- Data availability at each step
- File location structure
- Generation mode dispatch logic
- W&B metric grouping

---

### 4. VALIDATION_TEST_METRICS_ANALYSIS.md (605 lines)
**Complete reference with detailed analysis**

Consult this for:
- Deep understanding of each component
- Line-by-line code explanations
- Detailed metric computation logic
- Understanding metrics aggregation
- Information about W&B logging
- JSON persistence details
- Analysis tools reference

**Contains** (11 major sections):
1. Validation/Test loop flow (entry points, lifecycle)
2. Metrics computation location (detailed breakdown)
3. Metrics object & aggregation (metric classes)
4. Available sequence length information
5. Logging to Weights & Biases
6. File-based metrics persistence (JSON structures)
7. Analysis & reporting (analyze_metrics.py functions)
8. Key data flow diagram
9. Sequence length information (what's available vs. what's missing)
10. Summary table of pipeline components
11. Configuration reference

---

## Quick Navigation Matrix

| Need | Start With | Then Read |
|------|-----------|-----------|
| 5-min overview | ANALYSIS_SUMMARY | METRICS_QUICK_REFERENCE |
| Code location | METRICS_QUICK_REFERENCE | VALIDATION_TEST_METRICS_ANALYSIS |
| How it works | METRICS_ARCHITECTURE | VALIDATION_TEST_METRICS_ANALYSIS |
| Data flow | METRICS_ARCHITECTURE | VALIDATION_TEST_METRICS_ANALYSIS |
| Implementation | METRICS_QUICK_REFERENCE | METRICS_ARCHITECTURE |
| Deep dive | VALIDATION_TEST_METRICS_ANALYSIS | — |
| Missing features | ANALYSIS_SUMMARY | VALIDATION_TEST_METRICS_ANALYSIS |

---

## Key Concepts Summary

### Metrics Computed
- **Loss Metrics**: NLL, BPD, Perplexity
- **Accuracy Metrics**: exact match, token-level, final answer
- **Per Algorithm**: AR (1 mode), LT (1 mode), MDLM (5 modes)

### Where Metrics Go
- **Real-time**: Weights & Biases dashboard
- **Historical**: `validation_metrics.json` (cumulative), `test_metrics.json` (final)
- **Intermediate**: `callback_metrics` during epoch aggregation

### Computation Phases
1. **Per-batch**: Loss + accuracy per batch
2. **Epoch-end**: Aggregation + persistence

### Key Data Available (Not Currently Logged)
```python
seq_lengths = (batch["attention_mask"] == 1).sum(dim=1)
prompt_lengths = batch["do_not_mask"].sum(dim=1)
target_lengths = (~batch["do_not_mask"] & 
                  (batch["input_ids"] != pad_token_id)).sum(dim=1)
```

---

## Critical File Locations

| Task | File | Lines | Function |
|------|------|-------|----------|
| Main metrics class | metrics.py | 67-110 | `Metrics` class |
| Loss computation | trainer_base.py | 650-704 | `_loss()` |
| Metrics update | metrics.py | 103-109 | `update_valid()` |
| Validation step | trainer_base.py | 186-297 | `validation_step()` |
| Test step | trainer_base.py | 456-567 | `test_step()` |
| Epoch end (val) | trainer_base.py | 367-403 | `on_validation_epoch_end()` |
| Epoch end (test) | trainer_base.py | 569-601 | `on_test_epoch_end()` |
| Prompt split | trainer_base.py | 299-315 | `_extract_prompts_and_targets()` |
| Accuracy calc | trainer_base.py | 317-357 | `_compute_accuracy()` |
| AR generation | algo.py | 9-58 | `AR.generate_conditioned()` |
| LT generation | algo.py | 125-150 | `LT.generate_conditioned()` |
| Collator | masked_formal_collator.py | 8-85 | `MaskedFormalCollator` |
| Analysis tool | analyze_metrics.py | 30-351 | Various functions |

---

## Typical Questions & Answers

**Q: Where is the validation loss computed?**
A: In `trainer_base.py:193-200`, within `validation_step()`. See METRICS_QUICK_REFERENCE.md.

**Q: How are metrics aggregated across batches?**
A: Via `metrics.py:103-109`, which maintains running sums. See VALIDATION_TEST_METRICS_ANALYSIS.md Section 3.

**Q: Where does my data go?**
A: Two places: W&B dashboard (real-time) and JSON files. See METRICS_QUICK_REFERENCE.md "JSON File Structures".

**Q: What's the difference between acc_exact and acc_token?**
A: See ANALYSIS_SUMMARY.md "Three Core Metrics" or METRICS_QUICK_REFERENCE.md.

**Q: Can I analyze results by sequence length?**
A: Not currently. See ANALYSIS_SUMMARY.md "Missing Features" for implementation approach.

**Q: How do generation modes work?**
A: Different per algorithm. See METRICS_QUICK_REFERENCE.md "Generation Modes by Algorithm".

**Q: Where are the JSON files saved?**
A: `${checkpointing.save_dir}/`. See config.yaml and METRICS_ARCHITECTURE.md "Metrics File Locations".

---

## Document Statistics

| Document | Lines | Size | Focus |
|----------|-------|------|-------|
| ANALYSIS_SUMMARY | 245 | 7.8KB | Overview & navigation |
| METRICS_QUICK_REFERENCE | 210 | 6.0KB | Quick lookup |
| METRICS_ARCHITECTURE | 392 | 21KB | Visual diagrams |
| VALIDATION_TEST_METRICS_ANALYSIS | 605 | 19KB | Complete reference |
| **TOTAL** | **1,452** | **53.8KB** | **Comprehensive** |

---

## How to Use This Documentation

### For Quick Answers
1. Check ANALYSIS_SUMMARY.md "Key Takeaways"
2. If not found, search METRICS_QUICK_REFERENCE.md
3. For code locations, use the tables in any document

### For Understanding Implementation
1. Read METRICS_ARCHITECTURE.md relevant section
2. Get line numbers from METRICS_QUICK_REFERENCE.md
3. Examine code in source files
4. Cross-reference with VALIDATION_TEST_METRICS_ANALYSIS.md

### For Deep Dives
1. Start with VALIDATION_TEST_METRICS_ANALYSIS.md
2. Follow line number references to source code
3. Consult METRICS_ARCHITECTURE.md for context

### For Adding Features
1. Check ANALYSIS_SUMMARY.md "Missing Features"
2. Use METRICS_QUICK_REFERENCE.md code examples
3. Reference METRICS_ARCHITECTURE.md for where to add code
4. Implement following patterns in VALIDATION_TEST_METRICS_ANALYSIS.md

---

## Key Implementation Patterns

### Logging Metrics
```python
self.log(
    f"val/{gen_mode}_acc_token",
    acc_token,
    on_step=False,
    on_epoch=True,
    sync_dist=True
)
```

### Extracting Sequence Lengths
```python
seq_lengths = (batch["attention_mask"] == 1).sum(dim=1)
target_lengths = (~batch["do_not_mask"] & 
                  (batch["input_ids"] != pad_token_id)).sum(dim=1)
```

### Computing Accuracy
```python
target_mask = targets != self.tokenizer.pad_token_id
acc_exact = ((generated == targets) | ~target_mask).all(dim=1).float().mean()
acc_token = ((generated == targets) & target_mask).sum() / target_mask.sum()
```

### Saving to JSON
```python
current_metrics = {k: v.compute().item() for k, v in self.metrics.valid_nlls.items()}
current_metrics["epoch"] = self.current_epoch
current_metrics["global_step"] = self.global_step
# Append to list and save
```

---

## Related Source Files (Not Documented Here)

- `config.yaml` - Configuration system (referenced throughout)
- `dataloader.py` - Data loading (batch creation)
- `algo.py` - Algorithm implementations
- `diffusion.py` - Diffusion-specific logic
- `main.py` - Training entry point

These are referenced in the documentation but not analyzed in detail.

---

## Version Information

- Created: November 13, 2025
- Codebase: duo (research project)
- Framework: PyTorch Lightning
- Logging: Weights & Biases

---

## Quick Links to Source Code

- [trainer_base.py](trainer_base.py) - Primary metrics implementation
- [metrics.py](metrics.py) - Metric aggregation classes
- [masked_formal_collator.py](masked_formal_collator.py) - Batch structure
- [analyze_metrics.py](analyze_metrics.py) - Analysis tools
- [config.yaml](config.yaml) - Configuration
- [algo.py](algo.py) - Algorithm implementations

