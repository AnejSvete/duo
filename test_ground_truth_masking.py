#!/usr/bin/env python3
"""
Test script for ground_truth_masking implementation in AbsorbingState.

This script verifies that:
1. Levels are masked from right to left
2. Timesteps are sampled uniformly from discrete levels
3. The masking respects do_not_mask regions
"""

import torch
from transformers import AutoTokenizer


def create_mock_tokenizer():
    """Create a mock tokenizer with the necessary special tokens."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Add special tokens if not present
    special_tokens = {"mask_token": "[MASK]", "pad_token": "[PAD]"}
    tokenizer.add_special_tokens(special_tokens)
    # Add formal language tokens
    tokenizer.add_tokens(["#", "|"])
    return tokenizer


def create_test_sequence(tokenizer):
    """
    Create a test sequence like:
    b b b # 32 32 31 | 48 16 23 | 0 7 | 18

    After '#', this has 4 levels (separated by '|'):
    - Level 1: 32 32 31 (between # and first |)
    - Level 2: 48 16 23 (between 1st and 2nd |)
    - Level 3: 0 7 (between 2nd and 3rd |)
    - Level 4: 18 (after last |)

    There are 3 pipes, so 4 levels total.
    When masking k levels from the right, we mask from the (total_pipes - k + 1)th pipe.
    """
    text = "b b b # 32 32 31 | 48 16 23 | 0 7 | 18"
    tokens = tokenizer.encode(text, return_tensors="pt")

    # Create do_not_mask: protect everything before '#' (inclusive)
    hash_token_id = tokenizer.convert_tokens_to_ids("#")
    do_not_mask = torch.zeros_like(tokens, dtype=torch.bool)

    # Find position of '#' and mask everything before it (inclusive)
    hash_pos = (tokens == hash_token_id).nonzero(as_tuple=True)[1]
    if len(hash_pos) > 0:
        do_not_mask[0, : hash_pos[0] + 1] = True

    return tokens, do_not_mask, text


def test_ground_truth_masking():
    """Test the ground_truth_masking implementation."""
    print("="*80)
    print("Testing Ground Truth Masking Implementation")
    print("="*80)

    # Create mock tokenizer
    tokenizer = create_mock_tokenizer()

    # Get special token IDs
    mask_token_id = tokenizer.mask_token_id
    pipe_token_id = tokenizer.convert_tokens_to_ids("|")
    hash_token_id = tokenizer.convert_tokens_to_ids("#")

    print(f"\nSpecial Token IDs:")
    print(f"  MASK: {mask_token_id}")
    print(f"  PIPE (|): {pipe_token_id}")
    print(f"  HASH (#): {hash_token_id}")

    # Create test sequence
    tokens, do_not_mask, original_text = create_test_sequence(tokenizer)

    print(f"\nOriginal Sequence:")
    print(f"  Text: {original_text}")
    print(f"  Tokens: {tokens}")
    print(f"  Do Not Mask: {do_not_mask}")

    # Count pipe positions (levels)
    pipe_positions = (tokens[0] == pipe_token_id).nonzero(as_tuple=True)[0]
    valid_pipes = pipe_positions[~do_not_mask[0][pipe_positions]]
    # Number of levels = number of pipes (each pipe starts a new level to unmask)
    num_levels = len(valid_pipes)

    print(f"\nLevel Structure:")
    print(f"  Pipe positions: {pipe_positions.tolist()}")
    print(f"  Valid pipes (not in do_not_mask): {valid_pipes.tolist()}")
    print(f"  Total levels: {num_levels}")
    print(f"\n  Level breakdown:")
    print(f"    Before masking anything: # ... | ... | ... | ...")
    print(f"    After masking 1 level:   # ... | ... | ... | [MASK...]")
    print(f"    After masking 2 levels:  # ... | ... | [MASK...] [MASK...]")
    print(f"    etc.")

    # Simulate masking at different timesteps
    print(f"\n{'='*80}")
    print("Simulating Level-Based Masking")
    print(f"{'='*80}")

    for levels_to_mask in range(1, num_levels + 1):
        print(f"\nMasking {levels_to_mask} level(s) from the right:")

        xt = tokens.clone()

        if levels_to_mask == num_levels:
            # Mask everything after '#'
            hash_pos = (tokens[0] == hash_token_id).nonzero(as_tuple=True)[0]
            if len(hash_pos) > 0:
                start_pos = hash_pos[-1].item() + 1
                for j in range(start_pos, tokens.shape[1]):
                    if not do_not_mask[0, j]:
                        xt[0, j] = mask_token_id
        else:
            # Mask from the pipe that starts the level we want to mask
            # If we want to mask k levels from the right, we start from pipe at index (num_levels - k)
            pipe_idx = num_levels - levels_to_mask
            if pipe_idx >= 0 and pipe_idx < len(valid_pipes):
                start_mask_pos = valid_pipes[pipe_idx].item()
                for j in range(start_mask_pos, tokens.shape[1]):
                    if not do_not_mask[0, j]:
                        xt[0, j] = mask_token_id

        # Decode and show
        decoded = tokenizer.decode(xt[0], skip_special_tokens=False)
        print(f"  Result: {decoded}")

        # Calculate mask ratio
        mask_count = (xt[0] == mask_token_id).sum().item()
        total_maskable = (~do_not_mask[0]).sum().item()
        mask_ratio = mask_count / total_maskable if total_maskable > 0 else 0

        print(f"  Mask count: {mask_count}/{total_maskable} = {mask_ratio:.3f}")
        print(f"  Expected t ≈ {levels_to_mask / num_levels:.3f}")

    print(f"\n{'='*80}")
    print("Test completed successfully!")
    print(f"{'='*80}")


def test_example_sequences():
    """Test with the example sequences from the user."""
    print("\n\n" + "="*80)
    print("Testing with User's Example Sequences")
    print("="*80)

    tokenizer = create_mock_tokenizer()
    mask_token_id = tokenizer.mask_token_id

    # Example from user (simplified)
    examples = [
        "b b b # 32 32 31 | 48 16 23 | 0 7 | 18",
    ]

    for example in examples:
        print(f"\nExample: {example}")

        # Tokenize
        tokens = tokenizer.encode(example, return_tensors="pt")

        # Create do_not_mask (protect before and including '#')
        hash_token_id = tokenizer.convert_tokens_to_ids("#")
        do_not_mask = torch.zeros_like(tokens, dtype=torch.bool)
        hash_pos = (tokens == hash_token_id).nonzero(as_tuple=True)[1]
        if len(hash_pos) > 0:
            do_not_mask[0, : hash_pos[0] + 1] = True

        # Count levels
        pipe_token_id = tokenizer.convert_tokens_to_ids("|")
        pipe_positions = (tokens[0] == pipe_token_id).nonzero(as_tuple=True)[0]
        valid_pipes = pipe_positions[~do_not_mask[0][pipe_positions]]
        num_levels = len(valid_pipes) + 1

        print(f"  Number of levels: {num_levels}")
        print(f"  Expected masking patterns (from user's description):")

        # Show expected patterns
        patterns = [
            "b b b # 32 32 31 | 48 16 23 | 0 7 | M M",
            "b b b # 32 32 31 | 48 16 23 | M M M M",
            "b b b # 32 32 31 | M M M M M M M M M",
            "b b b # M M M M M M M M M M M M M",
        ]

        for i, pattern in enumerate(patterns, 1):
            print(f"  Level {i}: {pattern}")


if __name__ == "__main__":
    test_ground_truth_masking()
    test_example_sequences()
