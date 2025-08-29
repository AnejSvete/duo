# masked_formal_collator.py
from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer


class MaskedFormalCollator:
    """
    A data collator that tokenizes formal language traces and creates a
    boolean mask to protect the initial formula from being masked during training.

    The collator identifies the '#' symbol in each trace and generates a
    `do_not_mask` tensor, which is True for all tokens up to and including '#'
    and False for all subsequent tokens.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Get the token ID for the '#' separator.
        # It's crucial that '#' is a distinct token in your vocabulary.
        # If it's not, you may need to add it as a special token.
        self.hash_token_id = self.tokenizer.convert_tokens_to_ids("#")
        if self.hash_token_id == self.tokenizer.unk_token_id:
            print("Warning: '#' is not in the tokenizer vocabulary and maps to UNK.")
            print("Conditional masking may not work as intended.")

    def __call__(self, batch: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        """Processes a list of examples to create a batch tensor."""
        # The input `batch` is a list of dictionaries, e.g., [{'text': '...'}, ...]
        texts = [item["text"] for item in batch]

        print(f"texts: {texts[:10]}")

        tokenized_output = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        print(f"tokenized_output: {tokenized_output[:10]}")

        input_ids = tokenized_output["input_ids"]
        attention_mask = tokenized_output["attention_mask"]

        # Create the boolean tensor to specify which tokens to protect
        do_not_mask = torch.zeros_like(input_ids, dtype=torch.bool)

        for i, token_ids in enumerate(input_ids):
            # Find all occurrences of the '#' token
            hash_indices = (token_ids == self.hash_token_id).nonzero(as_tuple=True)[0]

            if len(hash_indices) > 0:
                # Protect all tokens up to and including the first '#'
                cutoff_index = hash_indices[0]
                do_not_mask[i, : cutoff_index + 1] = True
            else:
                # If a trace has no '#', protect the entire sequence as a fallback
                do_not_mask[i, :] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_not_mask": do_not_mask,
        }
