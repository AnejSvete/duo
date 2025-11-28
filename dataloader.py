import itertools
import os
import typing
from typing import Optional

import datasets
import tokenizers
import torch
import transformers

import arithmetic
import bfvp
import palindrome
import utils
from arithmetic import ARITHMETIC_CREATORS
from bfvp import BFVP_CREATORS
from masked_formal_collator import MaskedFormalCollator
from palindrome import PALINDROME_CREATORS
from regular import FSA_CREATORS, get_monoid_size, make_fsa_examples

LOGGER = utils.get_logger(__name__)


class FormalTokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        mask_token="[MASK]",
        language="bfvp",
        monoid_size: Optional[int] = None,
        num_vars: Optional[int] = None,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        format_mode: str = "trace",
        alphabet: Optional[list] = None,
        **kwargs,
    ):

        if language in BFVP_CREATORS:
            if num_vars is None:
                raise ValueError("num_vars must be provided for the bfvp language.")
            variable_tokens = []
            if format_mode == "lookup":
                variable_tokens = [f"x{i}" for i in range(1, num_vars + 1)]
            self.FORMAL_TOKENS = [
                "#",
                "|",
                "and",
                "or",
                "not",
                "T",
                "F",
            ] + variable_tokens
        elif language in FSA_CREATORS:
            if monoid_size is None:
                raise ValueError("monoid_size must be provided for FSA languages.")
            monoid_tokens = [str(i) for i in range(monoid_size)]
            self.FORMAL_TOKENS = ["#", "|", "a", "b"] + monoid_tokens
        elif language in ARITHMETIC_CREATORS:
            if num_vars is None or min_val is None or max_val is None:
                raise ValueError(
                    "num_vars, min_val, and max_val must be provided for the arithmetic language."
                )
            variable_tokens = []
            if format_mode == "lookup":
                variable_tokens = [f"x{i}" for i in range(1, num_vars + 1)]
            constant_tokens = [str(i) for i in range(min_val, max_val + 1)]
            self.FORMAL_TOKENS = (
                ["#", "|", "+", "-", "*", "/"] + variable_tokens + constant_tokens
            )
        elif language in PALINDROME_CREATORS:
            # Palindromes use alphabet symbols plus structural markers
            if alphabet is None:
                alphabet = ["a", "b"]  # Default alphabet

            # Base tokens
            base_tokens = ["#", "|", "$", "T", "F", "compare", "reverse", "len≠"]

            # Add alphabet symbols
            alphabet_tokens = list(alphabet)

            # Generate trace-specific tokens for each alphabet symbol
            trace_tokens = []
            if format_mode == "trace" or format_mode == "empty_trace":
                for symbol in alphabet:
                    trace_tokens.extend([
                        f"push_{symbol}",
                        f"pop={symbol}",
                        f"pop≠{symbol}",
                        f"skip_{symbol}",
                        f"empty≠{symbol}",
                    ])

            self.FORMAL_TOKENS = base_tokens + alphabet_tokens + trace_tokens
        else:
            raise ValueError(f"Unknown formal language: {language}")

        vocab = {pad_token: 0, bos_token: 1, eos_token: 2, mask_token: 3}
        offset = 4
        for i, tok in enumerate(self.FORMAL_TOKENS):
            vocab[tok] = i + offset
        self._vocab_str_to_int = vocab
        self._vocab_int_to_str = {v: k for k, v in vocab.items()}
        super().__init__(
            pad_token=pad_token,
            bos_token=bos_token,
            eos_token=eos_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return text.strip().split()

    def _convert_token_to_id(self, token: str) -> int:
        if token not in self._vocab_str_to_int:
            raise ValueError(f"Invalid token '{token}' for FormalTokenizer.")
        return self._vocab_str_to_int[token]

    def _convert_id_to_token(self, index: int) -> str:
        if index not in self._vocab_int_to_str:
            raise ValueError(f"Invalid token id '{index}' for FormalTokenizer.")
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int

    def build_inputs_with_special_tokens(
        self, token_ids_0: typing.List[int], token_ids_1: typing.Optional[typing.List[int]] = None
    ) -> typing.List[int]:
        """
        Build model inputs by prepending BOS token.

        Args:
            token_ids_0: List of token IDs for the first sequence
            token_ids_1: Optional list of token IDs for the second sequence (not used)

        Returns:
            List of token IDs with BOS prepended: [BOS] + token_ids_0
        """
        bos = [self.bos_token_id] if self.bos_token_id is not None else []
        if token_ids_1 is None:
            return bos + token_ids_0
        return bos + token_ids_0 + token_ids_1


def _group_texts(examples, block_size, bos, eos):
    concatenated_examples = list(itertools.chain(*examples["input_ids"]))
    total_length = len(concatenated_examples)
    new_block_size = block_size - 2
    total_length = (total_length // new_block_size) * new_block_size
    result, _values, _attn_masks = {}, [], []
    for i in range(0, total_length, new_block_size):
        _values.append([bos] + concatenated_examples[i : i + new_block_size] + [eos])
        _attn_masks.append(torch.ones(block_size))
    result["input_ids"], result["attention_mask"] = _values, _attn_masks
    return result


def _get_split_sizes(dataset_name, config):
    """Helper to get the sizes of all splits for disjoint dataset generation."""
    if dataset_name in BFVP_CREATORS:
        bfvp_cfg = getattr(config.data, "properties", {})
        train_size = getattr(bfvp_cfg, "num_examples_train", 50000)
        valid_size = getattr(bfvp_cfg, "num_examples_valid", 5000)
        test_size = getattr(bfvp_cfg, "num_examples_test", 5000)
    elif dataset_name in FSA_CREATORS:
        lang_cfg = getattr(config.data, "properties", {})
        train_size = getattr(lang_cfg, "num_examples_train", 50000)
        valid_size = getattr(lang_cfg, "num_examples_valid", 5000)
        test_size = getattr(lang_cfg, "num_examples_test", 5000)
    elif dataset_name in ARITHMETIC_CREATORS:
        arith_cfg = getattr(config.data, "properties", {})
        train_size = getattr(arith_cfg, "num_examples_train", 50000)
        valid_size = getattr(arith_cfg, "num_examples_valid", 5000)
        test_size = getattr(arith_cfg, "num_examples_test", 5000)
    elif dataset_name in PALINDROME_CREATORS:
        pal_cfg = getattr(config.data, "properties", {})
        train_size = getattr(pal_cfg, "num_examples_train", 50000)
        valid_size = getattr(pal_cfg, "num_examples_valid", 5000)
        test_size = getattr(pal_cfg, "num_examples_test", 5000)
    else:
        train_size, valid_size, test_size = 50000, 5000, 5000

    return {"train": train_size, "validation": valid_size, "test": test_size}


def _compute_dataset_statistics(split_pools, dataset_name):
    """
    Compute and print comprehensive statistics for all dataset splits.

    Args:
        split_pools: Dictionary with keys "train", "validation", "test" containing lists of examples
        dataset_name: Name of the dataset
    """
    import numpy as np

    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"DATASET STATISTICS: {dataset_name}")
    LOGGER.info(f"{'='*80}")

    for split_name in ["train", "validation", "test"]:
        examples = split_pools[split_name]
        texts = [ex["text"] for ex in examples]

        # Compute lengths (in tokens, space-separated)
        lengths = [len(text.split()) for text in texts]

        LOGGER.info(f"\n{split_name.upper()} Split:")
        LOGGER.info(f"  Number of examples: {len(examples)}")
        LOGGER.info(f"  Number of unique strings: {len(set(texts))}")

        # Length statistics
        if lengths:
            LOGGER.info("\n  Length Statistics (tokens):")
            LOGGER.info(f"    Min: {min(lengths)}")
            LOGGER.info(f"    Max: {max(lengths)}")
            LOGGER.info(f"    Mean: {np.mean(lengths):.2f}")
            LOGGER.info(f"    Median: {np.median(lengths):.2f}")
            LOGGER.info(f"    Std: {np.std(lengths):.2f}")

            # Quantiles
            quantiles = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            LOGGER.info("    Quantiles:")
            for q in quantiles:
                LOGGER.info(f"      {int(q*100)}%: {np.quantile(lengths, q):.2f}")

        # Label statistics (extract labels after "#" delimiter)
        labels = []
        for text in texts:
            if "#" in text:
                # Get the part after "#" and extract the final label
                completion_part = text.split("#")[1].strip()
                if "|" in completion_part:
                    # For trace format, get the last part after final "|"
                    final_label = completion_part.split("|")[-1].strip()
                else:
                    # For final_value format, the whole completion is the label
                    final_label = completion_part
                labels.append(final_label)

        if labels:
            unique_labels = sorted(set(labels))
            label_counts = {label: labels.count(label) for label in unique_labels}

            LOGGER.info("\n  Label Statistics:")
            LOGGER.info(f"    Number of unique labels: {len(unique_labels)}")
            LOGGER.info("    Label distribution:")

            # Show label proportions
            total = len(labels)
            for label in unique_labels:
                count = label_counts[label]
                proportion = count / total * 100
                LOGGER.info(f"      '{label}': {count} ({proportion:.2f}%)")

    LOGGER.info(f"\n{'='*80}\n")


def _generate_and_cache_all_splits(dataset_name, config, block_size, num_proc):
    """
    Generates ALL splits (train/validation/test) in a single pass to ensure disjoint datasets,
    then caches each split separately.
    """
    LOGGER.info(f"Generating ALL splits for {dataset_name} in a single pass...")

    # Get split sizes
    split_sizes = _get_split_sizes(dataset_name, config)
    seed = 42

    # Generate all splits at once in a single pass
    if dataset_name in BFVP_CREATORS:
        bfvp_cfg = getattr(config.data, "properties", {})
        num_vars = getattr(bfvp_cfg, "num_vars", 2)
        format_mode = getattr(bfvp_cfg, "format", "trace")

        # Get depth ranges per split
        min_train_depth = getattr(
            bfvp_cfg, "min_train_depth", getattr(bfvp_cfg, "min_depth", 1)
        )
        max_train_depth = getattr(
            bfvp_cfg, "max_train_depth", getattr(bfvp_cfg, "max_depth", 4)
        )
        min_val_depth = getattr(bfvp_cfg, "min_val_depth", min_train_depth)
        max_val_depth = getattr(bfvp_cfg, "max_val_depth", max_train_depth)
        min_test_depth = getattr(bfvp_cfg, "min_test_depth", min_train_depth)
        max_test_depth = getattr(bfvp_cfg, "max_test_depth", max_train_depth)

        depth_ranges = {
            "train": (min_train_depth, max_train_depth),
            "validation": (min_val_depth, max_val_depth),
            "test": (min_test_depth, max_test_depth),
        }

        # Get length ranges per split (for stratification)
        min_train_len = getattr(bfvp_cfg, "min_train_len", None)
        max_train_len = getattr(bfvp_cfg, "max_train_len", None)
        min_val_len = getattr(bfvp_cfg, "min_val_len", None)
        max_val_len = getattr(bfvp_cfg, "max_val_len", None)
        min_test_len = getattr(bfvp_cfg, "min_test_len", None)
        max_test_len = getattr(bfvp_cfg, "max_test_len", None)

        # Only create length_ranges dict if at least one length parameter is specified
        length_ranges = None
        if any(
            [
                min_train_len,
                max_train_len,
                min_val_len,
                max_val_len,
                min_test_len,
                max_test_len,
            ]
        ):
            length_ranges = {
                "train": (min_train_len or 0, max_train_len or 10000),
                "validation": (min_val_len or 0, max_val_len or 10000),
                "test": (min_test_len or 0, max_test_len or 10000),
            }

        # Get padding configuration
        padding_scale_type = getattr(bfvp_cfg, "padding_scale_type", "natural")
        padding_multiplier = getattr(bfvp_cfg, "padding_multiplier", 0.0)
        padding_constant = getattr(bfvp_cfg, "padding_constant", None)
        padding_max = getattr(bfvp_cfg, "padding_max", None)

        LOGGER.info(
            f"Generating bfvp data with: "
            f"train_depth=[{min_train_depth},{max_train_depth}], "
            f"val_depth=[{min_val_depth},{max_val_depth}], "
            f"test_depth=[{min_test_depth},{max_test_depth}], "
            f"num_vars={num_vars}, format={format_mode}, seed={seed}"
        )
        if length_ranges:
            LOGGER.info(
                f"  Length stratification: "
                f"train_len=[{min_train_len},{max_train_len}], "
                f"val_len=[{min_val_len},{max_val_len}], "
                f"test_len=[{min_test_len},{max_test_len}]"
            )
        if padding_multiplier > 0 or padding_constant:
            LOGGER.info(
                f"  Padding: scale_type={padding_scale_type}, multiplier={padding_multiplier}, "
                f"constant={padding_constant}, max={padding_max}"
            )

        split_pools = bfvp.make_all_splits(
            min_depth=min_train_depth,
            max_depth=max_train_depth,
            num_vars=num_vars,
            mode=format_mode,
            seed=seed,
            split_sizes=split_sizes,
            depth_ranges=depth_ranges,
            length_ranges=length_ranges,
            padding_scale_type=padding_scale_type,
            padding_multiplier=padding_multiplier,
            padding_constant=padding_constant,
            padding_max=padding_max,
        )

    elif dataset_name in FSA_CREATORS:
        lang_cfg = getattr(config.data, "properties", {})
        format_mode = getattr(lang_cfg, "format", "trace")

        # Get length ranges per split
        min_train_len = getattr(lang_cfg, "min_train_len", 32)
        max_train_len = getattr(lang_cfg, "max_train_len", 32)
        min_val_len = getattr(lang_cfg, "min_val_len", min_train_len)
        max_val_len = getattr(lang_cfg, "max_val_len", max_train_len)
        min_test_len = getattr(lang_cfg, "min_test_len", min_train_len)
        max_test_len = getattr(lang_cfg, "max_test_len", max_train_len)

        length_ranges = {
            "train": (min_train_len, max_train_len),
            "validation": (min_val_len, max_val_len),
            "test": (min_test_len, max_test_len),
        }

        # Get padding configuration
        padding_scale_type = getattr(lang_cfg, "padding_scale_type", "natural")
        padding_multiplier = getattr(lang_cfg, "padding_multiplier", 0.0)
        padding_constant = getattr(lang_cfg, "padding_constant", None)
        padding_max = getattr(lang_cfg, "padding_max", None)

        LOGGER.info(
            f"Generating {dataset_name} data with: "
            f"train_len=[{min_train_len},{max_train_len}], "
            f"val_len=[{min_val_len},{max_val_len}], "
            f"test_len=[{min_test_len},{max_test_len}], "
            f"format={format_mode}, seed={seed}"
        )
        if padding_multiplier > 0 or padding_constant:
            LOGGER.info(
                f"  Padding: scale_type={padding_scale_type}, multiplier={padding_multiplier}, "
                f"constant={padding_constant}, max={padding_max}"
            )

        fsa = FSA_CREATORS[dataset_name]()
        symbol_map, mult_table, identity_id, _, _ = fsa.compute_syntactic_monoid()
        monoid_details = {
            "symbol_map": symbol_map,
            "mult_table": mult_table,
            "identity_id": identity_id,
        }

        from regular import make_all_splits_fsa

        split_pools = make_all_splits_fsa(
            fsa=fsa,
            monoid_details=monoid_details,
            length_ranges=length_ranges,
            mode=format_mode,
            seed=seed,
            split_sizes=split_sizes,
            padding_scale_type=padding_scale_type,
            padding_multiplier=padding_multiplier,
            padding_constant=padding_constant,
            padding_max=padding_max,
        )

    elif dataset_name in ARITHMETIC_CREATORS:
        arith_cfg = getattr(config.data, "properties", {})
        min_val = getattr(arith_cfg, "min_val", 0)
        max_val = getattr(arith_cfg, "max_val", 50)
        format_mode = getattr(arith_cfg, "format", "trace")

        # Get depth ranges per split
        min_train_depth = getattr(
            arith_cfg, "min_train_depth", getattr(arith_cfg, "min_depth", 1)
        )
        max_train_depth = getattr(
            arith_cfg, "max_train_depth", getattr(arith_cfg, "max_depth", 4)
        )
        min_val_depth = getattr(arith_cfg, "min_val_depth", min_train_depth)
        max_val_depth = getattr(arith_cfg, "max_val_depth", max_train_depth)
        min_test_depth = getattr(arith_cfg, "min_test_depth", min_train_depth)
        max_test_depth = getattr(arith_cfg, "max_test_depth", max_train_depth)

        depth_ranges = {
            "train": (min_train_depth, max_train_depth),
            "validation": (min_val_depth, max_val_depth),
            "test": (min_test_depth, max_test_depth),
        }

        # Get length ranges per split (for stratification)
        min_train_len = getattr(arith_cfg, "min_train_len", None)
        max_train_len = getattr(arith_cfg, "max_train_len", None)
        min_val_len = getattr(arith_cfg, "min_val_len", None)
        max_val_len = getattr(arith_cfg, "max_val_len", None)
        min_test_len = getattr(arith_cfg, "min_test_len", None)
        max_test_len = getattr(arith_cfg, "max_test_len", None)

        # Only create length_ranges dict if at least one length parameter is specified
        length_ranges = None
        if any(
            [
                min_train_len,
                max_train_len,
                min_val_len,
                max_val_len,
                min_test_len,
                max_test_len,
            ]
        ):
            length_ranges = {
                "train": (min_train_len or 0, max_train_len or 10000),
                "validation": (min_val_len or 0, max_val_len or 10000),
                "test": (min_test_len or 0, max_test_len or 10000),
            }

        # Get padding configuration
        padding_scale_type = getattr(arith_cfg, "padding_scale_type", "natural")
        padding_multiplier = getattr(arith_cfg, "padding_multiplier", 0.0)
        padding_constant = getattr(arith_cfg, "padding_constant", None)
        padding_max = getattr(arith_cfg, "padding_max", None)

        LOGGER.info(
            f"Generating arithmetic data with: "
            f"train_depth=[{min_train_depth},{max_train_depth}], "
            f"val_depth=[{min_val_depth},{max_val_depth}], "
            f"test_depth=[{min_test_depth},{max_test_depth}], "
            f"min_val={min_val}, max_val={max_val}, format={format_mode}, seed={seed}"
        )
        if length_ranges:
            LOGGER.info(
                f"  Length stratification: "
                f"train_len=[{min_train_len},{max_train_len}], "
                f"val_len=[{min_val_len},{max_val_len}], "
                f"test_len=[{min_test_len},{max_test_len}]"
            )
        if padding_multiplier > 0 or padding_constant:
            LOGGER.info(
                f"  Padding: scale_type={padding_scale_type}, multiplier={padding_multiplier}, "
                f"constant={padding_constant}, max={padding_max}"
            )

        split_pools = arithmetic.make_all_splits(
            min_depth=min_train_depth,
            max_depth=max_train_depth,
            mode=format_mode,
            min_val=min_val,
            max_val=max_val,
            seed=seed,
            split_sizes=split_sizes,
            depth_ranges=depth_ranges,
            length_ranges=length_ranges,
            padding_scale_type=padding_scale_type,
            padding_multiplier=padding_multiplier,
            padding_constant=padding_constant,
            padding_max=padding_max,
        )

    elif dataset_name in PALINDROME_CREATORS:
        pal_cfg = getattr(config.data, "properties", {})
        format_mode = getattr(pal_cfg, "format", "final_value")
        alphabet_str = getattr(pal_cfg, "alphabet", "a,b")
        alphabet = alphabet_str.split(",")

        # Determine if this is marked or unmarked palindrome
        marked = dataset_name == "marked_palindrome"

        # Get length ranges per split
        min_train_len = getattr(pal_cfg, "min_train_len", 4)
        max_train_len = getattr(pal_cfg, "max_train_len", 32)
        min_val_len = getattr(pal_cfg, "min_val_len", min_train_len)
        max_val_len = getattr(pal_cfg, "max_val_len", max_train_len)
        min_test_len = getattr(pal_cfg, "min_test_len", min_train_len)
        max_test_len = getattr(pal_cfg, "max_test_len", max_train_len)

        length_ranges = {
            "train": (min_train_len, max_train_len),
            "validation": (min_val_len, max_val_len),
            "test": (min_test_len, max_test_len),
        }

        # Get padding configuration
        padding_scale_type = getattr(pal_cfg, "padding_scale_type", "natural")
        padding_multiplier = getattr(pal_cfg, "padding_multiplier", 0.0)
        padding_constant = getattr(pal_cfg, "padding_constant", None)
        padding_max = getattr(pal_cfg, "padding_max", None)

        LOGGER.info(
            f"Generating {dataset_name} data with: "
            f"train_len=[{min_train_len},{max_train_len}], "
            f"val_len=[{min_val_len},{max_val_len}], "
            f"test_len=[{min_test_len},{max_test_len}], "
            f"alphabet={alphabet}, format={format_mode}, seed={seed}"
        )
        if padding_multiplier > 0 or padding_constant:
            LOGGER.info(
                f"  Padding: scale_type={padding_scale_type}, multiplier={padding_multiplier}, "
                f"constant={padding_constant}, max={padding_max}"
            )

        split_pools = palindrome.make_all_splits(
            marked=marked,
            alphabet=alphabet,
            mode=format_mode,
            seed=seed,
            split_sizes=split_sizes,
            length_ranges=length_ranges,
            padding_scale_type=padding_scale_type,
            padding_multiplier=padding_multiplier,
            padding_constant=padding_constant,
            padding_max=padding_max,
        )

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Print dataset statistics after generation
    _compute_dataset_statistics(split_pools, dataset_name)

    # Now cache each split separately
    for mode in ["train", "validation", "test"]:
        examples = split_pools[mode]
        dataset = datasets.Dataset.from_list(examples)

        def preprocess_and_tokenize(examples):
            texts = examples["text"]
            tokenizer = get_tokenizer(config)
            tokenizer.padding_side, tokenizer.truncation_side = "right", "right"

            # Check for truncation issues (log first few examples)
            if hasattr(preprocess_and_tokenize, '_log_count'):
                preprocess_and_tokenize._log_count += 1
            else:
                preprocess_and_tokenize._log_count = 1

            if preprocess_and_tokenize._log_count <= 3:
                for i, text in enumerate(texts[:2]):
                    token_count = len(text.split())
                    LOGGER.warning(
                        f"Sample {i}: text has {token_count} tokens, "
                        f"max_length={block_size}, text='{text[:100]}...'"
                    )

            tokens = tokenizer(
                texts,
                max_length=block_size,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
            tokens["text"] = texts
            # Preserve label field if it exists (e.g., for palindrome tasks)
            if "label" in examples:
                tokens["label"] = examples["label"]
            return tokens

        tokenized_dataset = dataset.map(
            preprocess_and_tokenize,
            batched=True,
            num_proc=num_proc,
            load_from_cache_file=True,
            desc=f"Tokenizing {mode}",
        )

        # Get the cache path for this mode
        base_name = _get_base_name(dataset_name, config, mode)
        filename = f"{base_name}_{mode}_bs{block_size}.dat"
        _path = os.path.join(config.data.cache_dir, filename)

        tokenized_dataset.save_to_disk(_path)
        LOGGER.info(f"Saved {mode} dataset to: {_path}")


def _get_base_name(dataset_name, config, mode):
    """Helper to generate base name for cache files."""
    # Get common padding parameters
    props = getattr(config.data, "properties", {})
    padding_scale_type = getattr(props, "padding_scale_type", "linear")
    padding_multiplier = getattr(props, "padding_multiplier", 2.0)
    padding_constant = getattr(props, "padding_constant", None)
    padding_max = getattr(props, "padding_max", None)

    # Get dataset size for this mode
    if mode == "train":
        dataset_size = getattr(props, "num_examples_train", 50000)
    elif mode == "validation":
        dataset_size = getattr(props, "num_examples_valid", 5000)
    else:  # test
        dataset_size = getattr(props, "num_examples_test", 5000)

    # Build padding suffix (compact format)
    # Format: ps{type}_pm{mult}_pc{const}_px{max}
    padding_parts = [f"ps{padding_scale_type}"]
    if padding_multiplier != 2.0:  # Only include if non-default
        padding_parts.append(f"pm{padding_multiplier:.1f}".replace(".", "p"))
    if padding_constant is not None:
        padding_parts.append(f"pc{padding_constant}")
    if padding_max is not None:
        padding_parts.append(f"px{padding_max}")
    padding_suffix = "_".join(padding_parts)

    # Build size suffix (only if non-default)
    size_suffix = ""
    default_sizes = {"train": 50000, "validation": 5000, "test": 5000}
    if dataset_size != default_sizes.get(mode, 50000):
        size_suffix = f"_n{dataset_size}"

    if dataset_name in BFVP_CREATORS:
        bfvp_cfg = getattr(config.data, "properties", {})
        # Get mode-specific depth ranges
        if mode == "train":
            min_depth = getattr(
                bfvp_cfg, "min_train_depth", getattr(bfvp_cfg, "min_depth", 1)
            )
            max_depth = getattr(
                bfvp_cfg, "max_train_depth", getattr(bfvp_cfg, "max_depth", 4)
            )
        elif mode == "validation":
            default_min = getattr(
                bfvp_cfg, "min_train_depth", getattr(bfvp_cfg, "min_depth", 1)
            )
            default_max = getattr(
                bfvp_cfg, "max_train_depth", getattr(bfvp_cfg, "max_depth", 4)
            )
            min_depth = getattr(bfvp_cfg, "min_val_depth", default_min)
            max_depth = getattr(bfvp_cfg, "max_val_depth", default_max)
        else:  # test
            default_min = getattr(
                bfvp_cfg, "min_train_depth", getattr(bfvp_cfg, "min_depth", 1)
            )
            default_max = getattr(
                bfvp_cfg, "max_train_depth", getattr(bfvp_cfg, "max_depth", 4)
            )
            min_depth = getattr(bfvp_cfg, "min_test_depth", default_min)
            max_depth = getattr(bfvp_cfg, "max_test_depth", default_max)
        num_vars = getattr(bfvp_cfg, "num_vars", 2)
        format_str = getattr(bfvp_cfg, "format", "trace").replace("_", "-")
        return f"{dataset_name}_mind{min_depth}_maxd{max_depth}_nv{num_vars}_f-{format_str}_{padding_suffix}{size_suffix}"
    elif dataset_name in FSA_CREATORS:
        lang_cfg = getattr(config.data, "properties", {})
        # Get default training lengths first
        default_min = getattr(lang_cfg, "min_train_len", 32)
        default_max = getattr(lang_cfg, "max_train_len", 32)

        # Get mode-specific lengths with fallback to training defaults
        min_len = getattr(lang_cfg, f"min_{mode}_len", default_min)
        max_len = getattr(lang_cfg, f"max_{mode}_len", default_max)

        format_str = getattr(lang_cfg, "format", "trace").replace("_", "-")
        return f"{dataset_name}_minl{min_len}_maxl{max_len}_f-{format_str}_{padding_suffix}{size_suffix}"
    elif dataset_name in ARITHMETIC_CREATORS:
        arith_cfg = getattr(config.data, "properties", {})
        # Get mode-specific depth ranges
        if mode == "train":
            min_depth = getattr(
                arith_cfg, "min_train_depth", getattr(arith_cfg, "min_depth", 1)
            )
            max_depth = getattr(
                arith_cfg, "max_train_depth", getattr(arith_cfg, "max_depth", 4)
            )
        elif mode == "validation":
            default_min = getattr(
                arith_cfg, "min_train_depth", getattr(arith_cfg, "min_depth", 1)
            )
            default_max = getattr(
                arith_cfg, "max_train_depth", getattr(arith_cfg, "max_depth", 4)
            )
            min_depth = getattr(arith_cfg, "min_val_depth", default_min)
            max_depth = getattr(arith_cfg, "max_val_depth", default_max)
        else:  # test
            default_min = getattr(
                arith_cfg, "min_train_depth", getattr(arith_cfg, "min_depth", 1)
            )
            default_max = getattr(
                arith_cfg, "max_train_depth", getattr(arith_cfg, "max_depth", 4)
            )
            min_depth = getattr(arith_cfg, "min_test_depth", default_min)
            max_depth = getattr(arith_cfg, "max_test_depth", default_max)
        min_val = getattr(arith_cfg, "min_val", 0)
        max_val = getattr(arith_cfg, "max_val", 50)
        format_str = getattr(arith_cfg, "format", "trace").replace("_", "-")
        return f"{dataset_name}_mind{min_depth}_maxd{max_depth}_minv{min_val}_maxv{max_val}_f-{format_str}_{padding_suffix}{size_suffix}"
    elif dataset_name in PALINDROME_CREATORS:
        pal_cfg = getattr(config.data, "properties", {})
        # Get default training lengths first
        default_min = getattr(pal_cfg, "min_train_len", 4)
        default_max = getattr(pal_cfg, "max_train_len", 32)

        # Get mode-specific lengths with fallback to training defaults
        min_len = getattr(pal_cfg, f"min_{mode}_len", default_min)
        max_len = getattr(pal_cfg, f"max_{mode}_len", default_max)

        # Get alphabet info
        alphabet_str = getattr(pal_cfg, "alphabet", "a,b")
        alphabet_size = len(alphabet_str.split(","))

        format_str = getattr(pal_cfg, "format", "final_value").replace("_", "-")
        return f"{dataset_name}_minl{min_len}_maxl{max_len}_alph{alphabet_size}_f-{format_str}_{padding_suffix}{size_suffix}"
    else:
        return f"{dataset_name}_{padding_suffix}{size_suffix}"


def get_dataset(
    dataset_name,
    tokenizer,
    mode,
    cache_dir,
    block_size=1024,
    num_proc=len(os.sched_getaffinity(0)),
    config=None,
):
    # Check if ALL splits are cached
    all_modes = ["train", "validation", "test"]
    all_paths = {}
    for m in all_modes:
        base_name = _get_base_name(dataset_name, config, m)
        filename = f"{base_name}_{m}_bs{block_size}.dat"
        all_paths[m] = os.path.join(cache_dir, filename)

    # If any split is missing, regenerate ALL splits
    if not all(utils.fsspec_exists(p) for p in all_paths.values()):
        LOGGER.info(f"Cache miss. Generating all splits for {dataset_name}...")
        _generate_and_cache_all_splits(dataset_name, config, block_size, num_proc)
    else:
        LOGGER.info(f"Cache hit for all splits of {dataset_name}")

    # Load the requested split
    _path = all_paths[mode]
    LOGGER.info(f"Loading {mode} data from: {_path}")
    return datasets.load_from_disk(_path).with_format("torch")


def get_tokenizer(config):
    language = config.data.language
    monoid_size = None
    num_vars = None
    min_val = None
    max_val = None
    format_mode = "trace"
    alphabet = None
    # Pre-compute monoid size or num_vars for dynamic tokenizer vocab
    if language in BFVP_CREATORS:
        bfvp_cfg = getattr(config.data, "properties", {})
        num_vars = getattr(bfvp_cfg, "num_vars", 2)
        format_mode = getattr(bfvp_cfg, "format", "trace")
        LOGGER.info(
            f"Language '{language}' requires {num_vars} variables. Creating dynamic tokenizer."
        )
    elif language in FSA_CREATORS:
        monoid_size = get_monoid_size(language)
        LOGGER.info(
            f"Language '{language}' requires a monoid of size {monoid_size}. Creating dynamic tokenizer."
        )
    elif language in ARITHMETIC_CREATORS:
        arith_cfg = getattr(config.data, "properties", {})
        num_vars = 2  # Fixed to 2 variables for arithmetic
        min_val = getattr(arith_cfg, "min_val", 0)
        max_val = getattr(arith_cfg, "max_val", 50)
        format_mode = getattr(arith_cfg, "format", "trace")
        LOGGER.info(
            f"Language '{language}' requires 2 variables and values in [{min_val}, {max_val}]. Creating dynamic tokenizer."
        )
    elif language in PALINDROME_CREATORS:
        palindrome_cfg = getattr(config.data, "properties", {})
        alphabet_str = getattr(palindrome_cfg, "alphabet", "a,b")
        alphabet = alphabet_str.split(",")
        format_mode = getattr(palindrome_cfg, "format", "trace")
        LOGGER.info(
            f"Language '{language}' requires alphabet {alphabet}. Creating dynamic tokenizer."
        )
    tokenizer = FormalTokenizer(
        language=language,
        monoid_size=monoid_size,
        num_vars=num_vars,
        min_val=min_val,
        max_val=max_val,
        format_mode=format_mode,
        alphabet=alphabet,
    )

    if isinstance(
        tokenizer, (transformers.GPT2TokenizerFast, transformers.GPT2Tokenizer)
    ):
        tokenizer._tokenizer.post_processor = tokenizers.processors.BertProcessing(
            (tokenizer.bos_token, tokenizer.bos_token_id),
            (tokenizer.eos_token, tokenizer.eos_token_id),
        )

    if tokenizer.bos_token is None:
        tokenizer.bos_token = (
            tokenizer.cls_token if tokenizer.cls_token is not None else "[BOS]"
        )
    if tokenizer.eos_token is None:
        tokenizer.eos_token = (
            tokenizer.sep_token if tokenizer.sep_token is not None else "[EOS]"
        )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def get_dataloaders(
    config,
    tokenizer,
    skip_train=False,
    skip_valid=False,
    skip_test=False,
    valid_seed=None,
):
    num_gpus = torch.cuda.device_count()
    # Protect against zero GPUs (e.g., CPU-only machines) to avoid ZeroDivisionError.
    effective_num_gpus = max(1, num_gpus)
    effective_accum = max(1, int(config.trainer.accumulate_grad_batches))
    if config.loader.global_batch_size % (effective_num_gpus * effective_accum) != 0:
        raise ValueError(
            "Global batch size not divisible by number of GPUs and gradient accumulation steps."
        )
    if config.loader.eval_global_batch_size % effective_num_gpus != 0:
        raise ValueError("Eval batch size not divisible by number of GPUs.")

    train_set = (
        None
        if skip_train
        else get_dataset(
            config.data.language,
            tokenizer,
            mode="train",
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            num_proc=config.loader.num_workers,
            config=config,
        )
    )

    valid_set = (
        None
        if skip_valid
        else get_dataset(
            config.data.language,
            tokenizer,
            mode="validation",
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            num_proc=config.loader.num_workers,
            config=config,
        )
    )

    test_set = (
        None
        if skip_test
        else get_dataset(
            config.data.language,
            tokenizer,
            mode="test",
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            num_proc=config.loader.num_workers,
            config=config,
        )
    )

    collator = MaskedFormalCollator(tokenizer=tokenizer, max_length=config.model.length)
    train_loader, valid_loader, test_loader = None, None, None

    if not skip_train:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=config.loader.batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=True,
            persistent_workers=True,
            collate_fn=collator,
        )
        train_loader.tokenizer = tokenizer

    if not skip_valid:
        shuffle_valid = valid_seed is not None
        generator = torch.Generator().manual_seed(valid_seed) if shuffle_valid else None
        valid_loader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=shuffle_valid,
            generator=generator,
            collate_fn=collator,
        )
        valid_loader.tokenizer = tokenizer

    if not skip_test:
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=config.loader.eval_batch_size,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
            shuffle=False,
            collate_fn=collator,
        )
        test_loader.tokenizer = tokenizer

    return train_loader, valid_loader, test_loader
