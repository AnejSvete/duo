import argparse
import random
from typing import Dict, List


def generate_power_of_two_binary_string(max_log_len: int) -> str:
    """
    Generates a random binary string with a length that is a power of two.
    """
    if max_log_len <= 0:
        raise ValueError("max_log_len must be positive.")
    # Choose a random exponent to get a length like 2, 4, 8, etc.
    log_len = random.randint(1, max_log_len)
    length = 2**log_len
    return "".join(random.choice(["0", "1"]) for _ in range(length))


def get_parallel_parity_trace(binary_string: str) -> List[str]:
    """
    Computes the trace of parallel parity products for a given binary string.

    This shows the computation in layers, where each layer computes the
    parity of pairs from the previous layer.
    """
    # Convert the binary string to a list of integers (0s and 1s).
    # 0 represents even parity, 1 represents odd parity.
    current_level_parities = [int(char) for char in binary_string]

    trace_levels = []

    while len(current_level_parities) > 0:
        # Add the current level of partial parities to our trace.
        trace_levels.append(" ".join(map(str, current_level_parities)))

        # If we've reduced to a single value, the computation is done.
        if len(current_level_parities) == 1:
            break

        next_level_parities = []
        # Combine pairs of parities to produce the next level.
        # The XOR operation (^) is equivalent to parity addition.
        for i in range(0, len(current_level_parities), 2):
            combined_parity = current_level_parities[i] ^ current_level_parities[i + 1]
            next_level_parities.append(combined_parity)

        current_level_parities = next_level_parities

    return trace_levels


def make_parity_examples(
    num_examples: int, max_log_len: int, mode: str
) -> List[Dict[str, str]]:
    """
    Generates examples for the parity language based on the specified mode.
    """
    examples = []
    for _ in range(num_examples):
        binary_string = generate_power_of_two_binary_string(max_log_len)
        trace_levels = get_parallel_parity_trace(binary_string)

        # NEW: Create a space-separated version of the input string for the output.
        space_separated_string = " ".join(binary_string)

        if mode == "trace":
            # Join all computation levels with a separator.
            # The first level of the trace is the original string, so we can just use the trace.
            text = f"{' | '.join(trace_levels)}"
        elif mode == "final_value":
            # The final value is the last (and only) element of the last level.
            final_value = trace_levels[-1]
            text = f"{space_separated_string} # {final_value}"
        else:
            raise ValueError(f"Unknown format mode: {mode}")

        examples.append({"text": text})

    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate examples for the Parity language with parallel reduction."
    )
    parser.add_argument(
        "--max_log_len",
        type=int,
        default=4,
        help="The maximum power of two for string length (e.g., 4 -> max length 16).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="trace",
        choices=["trace", "final_value"],
        help="Output format: 'trace' for full reduction, 'final_value' for the result only.",
    )
    parser.add_argument(
        "--num_examples", type=int, default=10, help="Number of examples to generate."
    )
    parser.add_argument(
        "--show_examples", action="store_true", help="Print the generated examples."
    )

    args = parser.parse_args()

    print(
        f"Generating {args.num_examples} Parity examples with max log-length {args.max_log_len}."
    )
    print(f"Output format: '{args.format}'")

    examples = make_parity_examples(
        num_examples=args.num_examples,
        max_log_len=args.max_log_len,
        mode=args.format,
    )

    if args.show_examples:
        print("\n--- Generated Examples ---")
        for i, ex in enumerate(examples):
            print(f"[{i+1}] {ex['text']}")
