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
        texts = [item["text"] for item in batch]

        tokenized_output = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = tokenized_output["input_ids"]
        attention_mask = tokenized_output["attention_mask"]
        pad_token_id = self.tokenizer.pad_token_id

        # 1. Create a boolean mask for all '#' tokens in the batch
        # Shape: (batch_size, seq_len)
        is_hash = input_ids == self.hash_token_id

        # 2. Use cumsum to count hashes row-wise. A subsequent '#' will have a count > 1.
        # This creates a tensor where each element is the number of hashes seen so far in its row.
        hash_counts = is_hash.cumsum(dim=1)

        # 3. Create a mask for tokens that are a '#' AND have a count > 1
        mask_to_replace = is_hash & (hash_counts > 1)

        # 4. Apply the mask to replace subsequent '#' tokens and update the attention mask
        print(f"Text: {texts[:3]}")
        print(f"Input IDs before: {input_ids[:3]}")
        print(f"Is hash: {is_hash[:3]}")
        print(f"Hash counts: {hash_counts[:3]}")
        print(f"Mask to replace: {mask_to_replace[:3]}")
        input_ids[mask_to_replace] = pad_token_id
        print(f"Input IDs after: {input_ids[:3]}")
        attention_mask[mask_to_replace] = 0

        # 5. Create the 'do_not_mask' tensor in a vectorized way
        # Find the first '#' in each row. argmax returns the *first* True index.
        is_first_hash = is_hash & (hash_counts == 1)
        cutoff_indices = torch.argmax(is_first_hash.int(), dim=1)

        # Create a range tensor [0, 1, 2, ...] to compare against the cutoff indices
        col_indices = torch.arange(input_ids.shape[1], device=input_ids.device)

        # Create the mask by broadcasting. This is True for all positions up to the cutoff.
        do_not_mask = col_indices <= cutoff_indices.unsqueeze(1)

        print(f"Attention Mask: {attention_mask[:3]}")
        print(f"Do Not Mask: {do_not_mask[:3]}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "do_not_mask": do_not_mask,
        }
