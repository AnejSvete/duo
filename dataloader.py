import functools
import itertools
import json
import math
import os
import re
import shutil
import typing
import urllib
import zipfile
from typing import Optional

import datasets
import fsspec
import numpy as np
import requests
import tokenizers
import torch
import transformers

import arithmetic
import bfvp
import utils
from arithmetic import ARITHMETIC_CREATORS
from bfvp import BFVP_CREATORS
from masked_formal_collator import MaskedFormalCollator
from regular import FSA_CREATORS, get_monoid_size, make_fsa_examples

LOGGER = utils.get_logger(__name__)


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string


def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x


def lm1b_detokenizer(x):
    x = x.replace("http : / / ", "http://")
    x = x.replace("https : / / ", "https://")
    x = re.sub(r" \'(\w+)", r"'\1", x)
    x = re.sub(r" (\w+) \. ", r" \1. ", x)
    x = re.sub(r" (\w+) \.$", r" \1.", x)
    x = x.replace(" ? ", "? ")
    x = re.sub(r" \?$", "?", x)
    x = x.replace(" ! ", "! ")
    x = re.sub(r" \!$", "!", x)
    x = x.replace(" , ", ", ")
    x = x.replace(" : ", ": ")
    x = x.replace(" ; ", "; ")
    x = x.replace(" / ", "/")
    x = re.sub(r"\" ([^\"]+) \"", r'"\1"', x)
    x = re.sub(r"\' ([^\']+) \'", r"'\1'", x)
    x = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", x)
    x = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", x)
    x = x.replace("$ ", "$")
    x = x.replace("£ ", "£")
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return "\n" + text.strip()


def scientific_papers_detokenizer(x):
    x = wt_detokenizer(x)
    x = lm1b_detokenizer(x)
    return x


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


class SyntheticTokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        vocab_size,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token=None,
        cls_token=None,
        pad_token=None,
        mask_token=None,
        unk_token=None,
        **kwargs,
    ):
        self.tokens = []
        for i in range(vocab_size - 2):
            self.tokens.append(str(i) + " ")
        self._vocab_str_to_int = {
            "[BOS]": vocab_size - 2,
            "[EOS]": vocab_size - 1,
            **{ch: i for i, ch in enumerate(self.tokens)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def _generate_synthetic_data(dataset_size, seq_len, vocab_size):
    dataset = np.zeros((dataset_size, seq_len), dtype=int)
    dataset[:, 0] = vocab_size - 2
    dataset[:, -1] = vocab_size - 1
    for i in range(dataset_size):
        temp = np.random.randint(vocab_size - 2)
        for j in reversed(range(1, seq_len - 1)):
            dataset[i, j] = temp
            if temp != 0:
                temp = temp // 4
            else:
                temp = np.random.randint(vocab_size - 2)
    return dataset


def generate_synthetic_dataset(
    train_dataset_size, validation_dataset_size, seq_len, vocab_size
):
    np.random.seed(42)
    train_data = torch.from_numpy(
        _generate_synthetic_data(train_dataset_size, seq_len, vocab_size)
    )
    train_dataset = datasets.Dataset.from_dict(
        {"input_ids": train_data, "attention_mask": torch.ones_like(train_data)}
    )
    train_dataset.set_format(type="torch")
    np.random.seed(41)
    validation_data = torch.from_numpy(
        _generate_synthetic_data(validation_dataset_size, seq_len, vocab_size)
    )
    validation_dataset = datasets.Dataset.from_dict(
        {
            "input_ids": validation_data,
            "attention_mask": torch.ones_like(validation_data),
        }
    )
    validation_dataset.set_format(type="torch")
    return {"train": train_dataset, "validation": validation_dataset}


class Text8Tokenizer(transformers.PreTrainedTokenizer):
    def __init__(
        self,
        bos_token="[BOS]",
        eos_token="[EOS]",
        sep_token="[SEP]",
        cls_token="[CLS]",
        pad_token="[PAD]",
        mask_token="[MASK]",
        unk_token="[UNK]",
        **kwargs,
    ):
        self.characters = list("abcdefghijklmnopqrstuvwxyz ")
        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[EOS]": 3,
            "[MASK]": 4,
            "[PAD]": 5,
            "[RESERVED]": 6,
            "[UNK]": 7,
            **{ch: i + 8 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **kwargs) -> typing.List[str]:
        return list(text.lower())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_vocab(self) -> typing.Dict[str, int]:
        return self._vocab_str_to_int


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = [
            json.loads(line)
            for line in response.iter_lines(decode_unicode=True)
            if line
        ]
        return data_list

    return datasets.Dataset.from_list(read_jsonl_to_list(url))


def get_text8_dataset(cache_dir, max_seq_length=256, drop_last=True, crop_train=False):
    url = "http://mattmahoney.net/dc/text8.zip"
    cache_dir = (
        f"{cache_dir}/text8" if not crop_train else f"{cache_dir}/text8-crop-train"
    )
    split_names = ["train", "validation", "test"]
    if not all(
        [utils.fsspec_exists(os.path.join(cache_dir, split)) for split in split_names]
    ):
        raw_cache_dir = os.path.join(cache_dir, "raw_data")
        if not all(
            [
                utils.fsspec_exists(os.path.join(raw_cache_dir, f"text8.{split}.txt"))
                for split in split_names
            ]
        ):
            if not utils.fsspec_exists(os.path.join(raw_cache_dir, "text8.zip")):
                utils.fsspec_mkdirs(raw_cache_dir, exist_ok=True)
                LOGGER.info("Downloading text8 from URL {}.".format(url))
                with urllib.request.urlopen(url) as in_stream, open(
                    os.path.join(raw_cache_dir, "text8.zip"), "wb"
                ) as out_file:
                    shutil.copyfileobj(in_stream, out_file)
            with fsspec.open(os.path.join(raw_cache_dir, "text8.zip"), "rb") as f:
                rawdata = zipfile.ZipFile(f).read("text8").decode("utf-8")
            splits = {
                "train": rawdata[:90000000],
                "validation": rawdata[90000000:95000000],
                "test": rawdata[95000000:],
            }
            for split, data in splits.items():
                with fsspec.open(
                    os.path.join(raw_cache_dir, f"text8.{split}.txt"), "w"
                ) as f:
                    f.write(data)
        else:
            splits = {}
            for split in split_names:
                with fsspec.open(
                    os.path.join(raw_cache_dir, f"text8.{split}.txt"), "r"
                ) as f:
                    splits[split] = f.read()

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        dataset_dict = {}
        for k, v in splits.items():
            chunk_size = (
                2 * max_seq_length if k == "train" and crop_train else max_seq_length
            )
            text = list(chunks(v, chunk_size))
            if drop_last and len(text[-1]) < chunk_size:
                text = text[:-1]
            dataset_dict[k] = datasets.Dataset.from_dict({"text": text})
        dataset = datasets.DatasetDict(dataset_dict)
        dataset.save_to_disk(cache_dir)
    else:
        dataset = datasets.load_from_disk(cache_dir)
    return dataset


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


def get_dataset(
    dataset_name,
    tokenizer,
    wrap,
    mode,
    cache_dir,
    insert_eos=True,
    block_size=1024,
    num_proc=len(os.sched_getaffinity(0)),
    streaming=False,
    revision: Optional[str] = None,
    config=None,
):
    eos_tag = ""
    if not insert_eos:
        eos_tag = "_eosFalse"

    if dataset_name in BFVP_CREATORS:
        bfvp_cfg = getattr(config.data, "properties", {})
        min_depth = getattr(
            bfvp_cfg, "min_depth_train" if mode == "train" else "min_depth_valid", 1
        )
        max_depth = getattr(
            bfvp_cfg, "max_depth_train" if mode == "train" else "max_depth_valid", 3
        )
        num_vars = getattr(bfvp_cfg, "num_vars", 4)
        fan_in = getattr(bfvp_cfg, "fan_in", 2)
        format_str = getattr(bfvp_cfg, "format", "trace").replace("_", "-")
        base_name = f"{dataset_name}_mind{min_depth}_maxd{max_depth}_nv{num_vars}_fi{fan_in}_f-{format_str}"
    elif dataset_name in FSA_CREATORS:
        lang_cfg = getattr(config.data, "properties", {})
        min_len, max_len = getattr(lang_cfg, f"min_len_{mode}", 32), getattr(
            lang_cfg, f"max_len_{mode}", 32
        )
        format_str = getattr(lang_cfg, "format", "trace").replace("_", "-")
        base_name = f"{dataset_name}_minl{min_len}_maxl{max_len}_f-{format_str}"
    elif dataset_name in ARITHMETIC_CREATORS:
        arith_cfg = getattr(config.data, "properties", {})
        min_depth = getattr(
            arith_cfg, "min_depth_train" if mode == "train" else "min_depth_valid", 1
        )
        max_depth = getattr(
            arith_cfg, "max_depth_train" if mode == "train" else "max_depth_valid", 4
        )
        num_vars = getattr(arith_cfg, "num_vars", 2)
        min_val = getattr(arith_cfg, "min_val", 0)
        max_val = getattr(arith_cfg, "max_val", 50)
        format_str = getattr(arith_cfg, "format", "trace").replace("_", "-")
        base_name = f"{dataset_name}_mind{min_depth}_maxd{max_depth}_nv{num_vars}_minv{min_val}_maxv{max_val}_f-{format_str}"
    else:
        base_name = dataset_name

    if wrap:
        filename = f"{base_name}_{mode}_bs{block_size}_wrapped{eos_tag}.dat"
    else:
        filename = f"{base_name}_{mode}_bs{block_size}_unwrapped{eos_tag}.dat"
    _path = os.path.join(cache_dir, filename)
    if utils.fsspec_exists(_path):
        LOGGER.info(f"Loading data from: {_path}")
        return datasets.load_from_disk(_path).with_format("torch")
    LOGGER.info(f"Generating new data at: {_path}")

    crop_train = dataset_name == "text8-crop"
    if mode == "train" and crop_train:
        block_size *= 2

    if dataset_name in BFVP_CREATORS:
        bfvp_cfg = getattr(config.data, "properties", {})
        num_examples = (
            getattr(bfvp_cfg, "num_examples_train", 50000)
            if mode == "train"
            else getattr(bfvp_cfg, "num_examples_valid", 5000)
        )
        split_name = "train" if mode == "train" else "validation"
        min_depth = getattr(
            bfvp_cfg, "min_depth_train" if mode == "train" else "min_depth_valid", 1
        )
        max_depth = getattr(
            bfvp_cfg, "max_depth_train" if mode == "train" else "max_depth_valid", 3
        )
        num_vars, fan_in = (
            getattr(bfvp_cfg, "num_vars", 4),
            getattr(bfvp_cfg, "fan_in", 2),
        )
        format_mode = getattr(bfvp_cfg, "format", "trace")
        LOGGER.info(
            f"Generating '{split_name}' bfvp data with: min_depth={min_depth}, max_depth={max_depth}, num_vars={num_vars}, fan_in={fan_in}, format={format_mode}"
        )
        examples = bfvp.make_examples(
            num_examples=num_examples,
            min_depth=min_depth,
            max_depth=max_depth,
            num_vars=num_vars,
            fan_in=fan_in,
            mode=format_mode,
        )
        dataset = datasets.DatasetDict(
            {split_name: datasets.Dataset.from_list(examples)}
        )
    elif dataset_name in FSA_CREATORS:
        lang_cfg = getattr(config.data, "properties", {})
        num_examples = getattr(lang_cfg, f"num_examples_{mode}", 50000)
        split_name = "train" if mode == "train" else "validation"
        min_len, max_len = getattr(lang_cfg, f"min_len_{mode}", 32), getattr(
            lang_cfg, f"max_len_{mode}", 32
        )
        format_mode = getattr(lang_cfg, "format", "trace")
        LOGGER.info(f"Generating '{split_name}' {dataset_name} data...")
        fsa = FSA_CREATORS[dataset_name]()
        symbol_map, mult_table, identity_id, _, _ = fsa.compute_syntactic_monoid()
        monoid_details = {
            "symbol_map": symbol_map,
            "mult_table": mult_table,
            "identity_id": identity_id,
        }
        examples = make_fsa_examples(
            fsa,
            monoid_details,
            num_examples,
            min_len,
            max_len,
            format_mode,
        )
        dataset = datasets.DatasetDict(
            {split_name: datasets.Dataset.from_list(examples)}
        )
    elif dataset_name in ARITHMETIC_CREATORS:
        arith_cfg = getattr(config.data, "properties", {})
        num_examples = (
            getattr(arith_cfg, "num_examples_train", 50000)
            if mode == "train"
            else getattr(arith_cfg, "num_examples_valid", 5000)
        )
        split_name = "train" if mode == "train" else "validation"
        min_depth = getattr(
            arith_cfg, "min_depth_train" if mode == "train" else "min_depth_valid", 1
        )
        max_depth = getattr(
            arith_cfg, "max_depth_train" if mode == "train" else "max_depth_valid", 4
        )
        num_vars = getattr(arith_cfg, "num_vars", 2)
        min_val = getattr(arith_cfg, "min_val", 0)
        max_val = getattr(arith_cfg, "max_val", 50)
        format_mode = getattr(arith_cfg, "format", "trace")
        LOGGER.info(
            f"Generating '{split_name}' arithmetic data with: min_depth={min_depth}, max_depth={max_depth}, "
            f"num_vars={num_vars}, min_val={min_val}, max_val={max_val}, format={format_mode}"
        )
        examples = arithmetic.make_examples(
            num_examples=num_examples,
            min_depth=min_depth,
            max_depth=max_depth,
            mode=format_mode,
            min_val=min_val,
            max_val=max_val,
            num_vars=num_vars,
        )
        dataset = datasets.DatasetDict(
            {split_name: datasets.Dataset.from_list(examples)}
        )
    elif dataset_name == "wikitext103":
        dataset = datasets.load_dataset(
            "wikitext",
            name="wikitext-103-raw-v1",
            cache_dir=cache_dir,
            revision=revision,
        )
    elif dataset_name == "wikitext2":
        dataset = datasets.load_dataset(
            "wikitext", name="wikitext-2-raw-v1", cache_dir=cache_dir, revision=revision
        )
    elif dataset_name == "ptb":
        dataset = datasets.load_dataset(
            "ptb_text_only", cache_dir=cache_dir, revision=revision
        )
    elif dataset_name == "lambada":
        dataset = get_lambada_test_dataset()
    elif dataset_name == "text8":
        dataset = get_text8_dataset(cache_dir, max_seq_length=block_size)
    elif dataset_name == "text8-crop":
        dataset = get_text8_dataset(
            cache_dir, max_seq_length=block_size, crop_train=True
        )
    elif dataset_name == "openwebtext-train":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[:-100000]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
        )
    elif dataset_name == "openwebtext-valid":
        dataset = datasets.load_dataset(
            "openwebtext",
            split="train[-100000:]",
            cache_dir=cache_dir,
            revision=revision,
            streaming=False,
            num_proc=num_proc,
        )
    elif dataset_name == "scientific_papers_arxiv":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "arxiv",
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
    elif dataset_name == "scientific_papers_pubmed":
        dataset = datasets.load_dataset(
            "scientific_papers",
            "pubmed",
            cache_dir=cache_dir,
            streaming=streaming,
            revision=revision,
        )
    elif dataset_name == "ag_news":
        dataset = datasets.load_dataset(
            "ag_news", cache_dir=cache_dir, streaming=streaming, revision=revision
        )
    elif dataset_name == "synthetic":
        dataset = generate_synthetic_dataset(
            train_dataset_size=100000,
            validation_dataset_size=1024,
            seq_len=32,
            vocab_size=256,
        )
    else:
        dataset = datasets.load_dataset(
            dataset_name, cache_dir=cache_dir, streaming=streaming, revision=revision
        )

    data = (
        dataset
        if dataset_name in ["lambada", "openwebtext-train", "openwebtext-valid"]
        else dataset[mode]
    )
    if dataset_name == "synthetic":
        return data

    detokenizer = None
    if dataset_name.startswith("wikitext"):
        detokenizer = wt_detokenizer
    elif dataset_name == "ptb":
        detokenizer = ptb_detokenizer
    elif dataset_name == "lm1b":
        detokenizer = lm1b_detokenizer
    elif dataset_name == "lambada":
        detokenizer = lambada_detokenizer
    elif dataset_name.startswith("scientific_papers"):
        detokenizer = scientific_papers_detokenizer

    EOS, BOS = (
        tokenizer.encode(tokenizer.eos_token)[0],
        tokenizer.encode(tokenizer.bos_token)[0],
    )

    def preprocess_and_tokenize(example):
        text = (
            example["sentence"]
            if dataset_name == "ptb"
            else example.get("article", example["text"])
        )
        if detokenizer:
            text = [detokenizer(t) for t in text]
        tokenizer.padding_side, tokenizer.truncation_side = "right", "right"
        if wrap:
            tokens = tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            if insert_eos:
                tokens = {"input_ids": [t + [EOS] for t in tokens["input_ids"]]}
        else:
            tokens = tokenizer(
                text,
                max_length=block_size,
                padding="max_length",
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
            )
        return tokens

    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Tokenizing",
    )

    remove_cols = []
    if dataset_name == "ptb":
        remove_cols = ["sentence"]
    elif "scientific_papers" in dataset_name:
        remove_cols = ["article", "abstract", "section_names"]
    elif dataset_name == "ag_news":
        remove_cols = ["text", "label"]
    elif (
        dataset_name in BFVP_CREATORS
        or dataset_name in FSA_CREATORS
        or dataset_name in ARITHMETIC_CREATORS
    ):
        remove_cols = ["text"] if "text" in tokenized_dataset.column_names else []
    else:
        remove_cols = ["text"]
    if remove_cols:
        tokenized_dataset = tokenized_dataset.remove_columns(remove_cols)

    if not wrap:
        if (
            dataset_name in BFVP_CREATORS
            or dataset_name in FSA_CREATORS
            or dataset_name in ARITHMETIC_CREATORS
        ):
            original_texts = (
                data["text"]
                if isinstance(data, datasets.Dataset)
                else data[mode]["text"]
            )
            n = len(tokenized_dataset)
            if len(original_texts) > n:
                original_texts = original_texts[:n]
            elif len(original_texts) < n:
                original_texts += [original_texts[-1]] * (n - len(original_texts))
            tokenized_dataset = tokenized_dataset.add_column("text", original_texts)
        if not streaming:
            tokenized_dataset.save_to_disk(_path)
        return tokenized_dataset.with_format("torch")

    group_texts = functools.partial(
        _group_texts, block_size=block_size, bos=BOS, eos=EOS
    )
    chunked_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        load_from_cache_file=True,
        desc="Grouping",
    )

    if (
        dataset_name in BFVP_CREATORS
        or dataset_name in FSA_CREATORS
        or dataset_name in ARITHMETIC_CREATORS
    ):
        original_texts = (
            data["text"] if isinstance(data, datasets.Dataset) else data[mode]["text"]
        )
        n = len(chunked_dataset)
        if len(original_texts) > n:
            original_texts = original_texts[:n]
        elif len(original_texts) < n:
            original_texts += [original_texts[-1]] * (n - len(original_texts))
        chunked_dataset = chunked_dataset.add_column("text", original_texts)

    if not streaming:
        chunked_dataset.save_to_disk(_path)
    return chunked_dataset.with_format("torch")


def get_tokenizer(config):
    if config.data.tokenizer_name_or_path == "formal":
        language = config.data.train
        monoid_size = None
        num_vars = None
        min_val = None
        max_val = None
        format_mode = "trace"
        # Pre-compute monoid size or num_vars for dynamic tokenizer vocab
        if language in BFVP_CREATORS:
            bfvp_cfg = getattr(config.data, "properties", {})
            num_vars = getattr(bfvp_cfg, "num_vars", 4)
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
            num_vars = getattr(arith_cfg, "num_vars", 2)
            min_val = getattr(arith_cfg, "min_val", 0)
            max_val = getattr(arith_cfg, "max_val", 50)
            format_mode = getattr(arith_cfg, "format", "trace")
            LOGGER.info(
                f"Language '{language}' requires {num_vars} variables and values in [{min_val}, {max_val}]. Creating dynamic tokenizer."
            )
        tokenizer = FormalTokenizer(
            language=language,
            monoid_size=monoid_size,
            num_vars=num_vars,
            min_val=min_val,
            max_val=max_val,
            format_mode=format_mode,
        )
    elif config.data.tokenizer_name_or_path == "text8":
        tokenizer = Text8Tokenizer()
    elif config.data.tokenizer_name_or_path == "bert-base-uncased":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif config.data.tokenizer_name_or_path == "synthetic":
        tokenizer = SyntheticTokenizer(vocab_size=256)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            config.data.tokenizer_name_or_path
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
    if (
        config.loader.global_batch_size
        % (num_gpus * config.trainer.accumulate_grad_batches)
        != 0
    ):
        raise ValueError(
            "Global batch size not divisible by number of GPUs and gradient accumulation steps."
        )
    if config.loader.eval_global_batch_size % num_gpus != 0:
        raise ValueError("Eval batch size not divisible by number of GPUs.")

    train_set = (
        None
        if skip_train
        else get_dataset(
            config.data.train,
            tokenizer,
            mode="train",
            wrap=config.data.wrap,
            insert_eos=config.data.insert_train_eos,
            cache_dir=config.data.cache_dir,
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=config.loader.num_workers,
            revision=config.data.get("train_revision"),
            config=config,
        )
    )

    validation_split = (
        "test" if config.data.valid in ["text8", "lm1b", "ag_news"] else "validation"
    )
    valid_set = (
        None
        if skip_valid
        else get_dataset(
            config.data.valid,
            tokenizer,
            wrap=config.data.wrap,
            mode=validation_split,
            cache_dir=config.data.cache_dir,
            insert_eos=config.data.insert_valid_eos,
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=config.loader.num_workers,
            revision=config.data.get("valid_revision"),
            config=config,
        )
    )

    test_split = "test"
    test_set = (
        None
        if skip_test
        else get_dataset(
            config.data.test,
            tokenizer,
            wrap=config.data.wrap,
            mode=test_split,
            cache_dir=config.data.cache_dir,
            insert_eos=config.data.insert_test_eos,
            block_size=config.model.length,
            streaming=config.data.streaming,
            num_proc=config.loader.num_workers,
            revision=config.data.get("test_revision"),
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
            shuffle=not config.data.streaming,
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
