import argparse
import random
from typing import Dict, List, Optional, Tuple

# Palindrome language creators dictionary
PALINDROME_CREATORS = {
    "marked_palindrome": True,
    "unmarked_palindrome": True,
}


def compute_extra_padding_length(
    input_length: int,
    natural_trace_length: int,
    scale_type: str = "natural",
    multiplier: float = 0.0,
    constant: Optional[int] = None,
    max_length: Optional[int] = None,
) -> int:
    """
    Computes the amount of EXTRA empty padding to add beyond natural trace.

    Args:
        input_length: Length of the input sequence (before '#')
        natural_trace_length: Natural length of computation trace
        scale_type: Base quantity to scale ("natural", "linear", "quadratic", "cubic", "constant")
        multiplier: Scaling factor (0.0 = no extra padding)
        constant: Fixed padding length (for constant mode)
        max_length: Maximum extra padding length

    Returns:
        Number of additional [PAD] tokens to append
    """
    if multiplier == 0.0 and constant is None:
        return 0  # No extra padding

    if scale_type == "constant":
        extra = constant if constant is not None else 0
    elif scale_type == "natural":
        # Extra padding proportional to natural trace length
        extra = int(natural_trace_length * multiplier)
    elif scale_type == "linear":
        # Extra padding proportional to input length
        extra = int(input_length * multiplier)
    elif scale_type == "quadratic":
        extra = int((input_length**2) * multiplier)
    elif scale_type == "cubic":
        extra = int((input_length**3) * multiplier)
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    # Apply cap
    if max_length is not None:
        extra = min(extra, max_length)

    return max(0, extra)  # Never negative


def generate_palindrome(length: int, alphabet: List[str], marked: bool = True) -> str:
    """
    Generates a palindrome string over the given alphabet.

    Args:
        length: Length of the palindrome (length of w in both marked and unmarked)
        alphabet: List of symbols to use (e.g., ['a', 'b'])
        marked: If True, generates marked palindrome (w$w^R). If False, generates ww^R.

    Returns:
        Generated palindrome string
    """
    if length <= 0:
        raise ValueError("Length must be positive.")

    if marked:
        # For marked palindromes: generate w, then marker ($), then w^R (full reverse of w)
        # Length is the length of w
        w = [random.choice(alphabet) for _ in range(length)]
        w_reverse = w[::-1]
        return " ".join(w + ["$"] + w_reverse)
    else:
        # For unmarked palindromes: generate ww^R where the string reads same forwards/backwards
        # Length is the total length of the final palindrome
        half_len = length // 2
        first_half = [random.choice(alphabet) for _ in range(half_len)]

        # For even-length palindromes
        if length % 2 == 0:
            reverse_half = first_half[::-1]
            return " ".join(first_half + reverse_half)
        else:
            # For odd-length palindromes, add middle element
            middle = [random.choice(alphabet)]
            reverse_half = first_half[::-1]
            return " ".join(first_half + middle + reverse_half)


def generate_non_palindrome(length: int, alphabet: List[str], marked: bool = True) -> str:
    """
    Generates a NON-palindrome string over the given alphabet.

    Args:
        length: Length parameter (length of w for marked, total length for unmarked)
        alphabet: List of symbols to use (e.g., ['a', 'b'])
        marked: If True, generates marked non-palindrome (w$w'). If False, generates non-palindrome string.

    Returns:
        Generated non-palindrome string
    """
    if length <= 0:
        raise ValueError("Length must be positive.")

    if len(alphabet) < 2:
        raise ValueError("Alphabet must have at least 2 symbols to generate non-palindromes.")

    max_attempts = 100
    for _ in range(max_attempts):
        if marked:
            # Generate w and w', ensuring w' is NOT the reverse of w
            w = [random.choice(alphabet) for _ in range(length)]
            w_prime = [random.choice(alphabet) for _ in range(length)]

            # Ensure at least one position differs from the reverse
            w_reverse = w[::-1]
            if w_prime != w_reverse:
                return " ".join(w + ["$"] + w_prime)

            # Force a difference at a random position
            diff_pos = random.randint(0, length - 1)
            current = w_prime[diff_pos]
            alternatives = [s for s in alphabet if s != current]
            if alternatives:
                w_prime[diff_pos] = random.choice(alternatives)
                return " ".join(w + ["$"] + w_prime)
        else:
            # Generate a string that is NOT a palindrome
            symbols = [random.choice(alphabet) for _ in range(length)]

            # Check if it's accidentally a palindrome
            if symbols != symbols[::-1]:
                return " ".join(symbols)

            # Force it to be non-palindrome by changing a position
            # Change a position that will break symmetry
            change_pos = random.randint(0, length // 2)
            current = symbols[change_pos]
            alternatives = [s for s in alphabet if s != current]
            if alternatives:
                symbols[change_pos] = random.choice(alternatives)
                return " ".join(symbols)

    raise RuntimeError(f"Failed to generate non-palindrome after {max_attempts} attempts.")


def check_palindrome(input_string: str, marked: bool = True) -> bool:
    """
    Verifies if a string is a valid palindrome.

    Args:
        input_string: Space-separated string to check
        marked: If True, expects marked palindrome (w$w^R). If False, expects ww^R.

    Returns:
        True if valid palindrome, False otherwise
    """
    symbols = input_string.strip().split()

    if marked:
        # Check for marker ($)
        if "$" not in symbols:
            return False

        marker_idx = symbols.index("$")
        w = symbols[:marker_idx]
        w_reverse_actual = symbols[marker_idx + 1 :]

        # For marked palindromes: check if second half is exact reverse of w
        w_reverse_expected = w[::-1]
        return w_reverse_actual == w_reverse_expected
    else:
        # For unmarked palindromes: entire string should be a palindrome
        return symbols == symbols[::-1]


def get_palindrome_trace(
    input_string: str,
    marked: bool = True,
    mode: str = "trace",
    padding_scale_type: str = "natural",
    padding_multiplier: float = 0.0,
    padding_constant: Optional[int] = None,
    padding_max: Optional[int] = None,
) -> str:
    """
    Generates a trace showing palindrome verification.

    For marked palindromes (trace mode): w $ w^R # pos_0 | pos_1 | ... | result
    For unmarked palindromes (trace mode): ww^R # pos_0 | pos_1 | ... | result

    Args:
        input_string: The palindrome string to trace
        marked: Whether this is a marked palindrome
        mode: Format mode (trace, final_value, or verify)
        padding_scale_type: Type of padding scaling ("natural", "linear", "quadratic", "cubic", "constant")
        padding_multiplier: Multiplier for extra padding
        padding_constant: Fixed amount of extra padding (for constant mode)
        padding_max: Maximum extra padding length

    Returns:
        Formatted trace string
    """
    is_palindrome = check_palindrome(input_string, marked)
    print(f"Checking palindrome: '{input_string}' -> {'T' if is_palindrome else 'F'}")
    result = "T" if is_palindrome else "F"

    if mode == "final_value":
        # Just input and result (# separates prompt from completion)
        return f"{input_string} # {result}"

    elif mode == "trace":
        # Stack-based trace mimicking pushdown automaton (PDA)
        symbols = input_string.strip().split()

        if marked:
            marker_idx = symbols.index("$") if "$" in symbols else -1
            if marker_idx == -1:
                return f"{input_string} # F"

            w = symbols[:marker_idx]
            w_reverse = symbols[marker_idx + 1 :]
            input_length = len(w) + len(w_reverse)  # Exclude marker from input count

            # Phase 1: Push w onto stack
            push_steps = [f"push_{char}" for char in w]

            # Phase 2: Pop and compare with w^R
            pop_steps = []
            stack = w.copy()  # Simulate stack

            for i, char in enumerate(w_reverse):
                if i < len(stack):
                    expected = stack[len(stack) - 1 - i]
                    if char == expected:
                        pop_steps.append(f"pop={char}")
                    else:
                        pop_steps.append(f"pop≠{char}")
                        # Early termination on mismatch
                        all_steps = push_steps + pop_steps + ["F"]
                        return f"{input_string} # {' | '.join(all_steps)}"
                else:
                    # More chars in w_reverse than in stack
                    pop_steps.append(f"empty≠{char}")
                    all_steps = push_steps + pop_steps + ["F"]
                    return f"{input_string} # {' | '.join(all_steps)}"

            # Check if we popped everything (lengths match)
            if len(w) != len(w_reverse):
                all_steps = push_steps + pop_steps + ["len≠", "F"]
                return f"{input_string} # {' | '.join(all_steps)}"
            else:
                # Success case - add extra padding if configured
                trace_steps = push_steps + pop_steps
                natural_trace_length = len(trace_steps)

                # Compute extra padding
                extra_padding_count = compute_extra_padding_length(
                    input_length=input_length,
                    natural_trace_length=natural_trace_length,
                    scale_type=padding_scale_type,
                    multiplier=padding_multiplier,
                    constant=padding_constant,
                    max_length=padding_max,
                )

                if extra_padding_count > 0:
                    # Natural trace + extra padding + final
                    extra_padding_part = " ".join(["[PAD]"] * extra_padding_count)
                    all_steps = trace_steps + [extra_padding_part, "T"]
                    return f"{input_string} # {' | '.join(all_steps)}"
                else:
                    # Just natural trace (backward compatible)
                    all_steps = trace_steps + ["T"]
                    return f"{input_string} # {' | '.join(all_steps)}"

        else:
            # For unmarked palindromes: push first half, then pop and compare second half
            steps = []
            n = len(symbols)
            mid = n // 2
            input_length = n

            # Phase 1: Push first half onto stack
            push_steps = [f"push_{symbols[i]}" for i in range(mid)]

            # Handle odd-length palindromes (skip middle element)
            if n % 2 == 1:
                push_steps.append(f"skip_{symbols[mid]}")
                start_compare = mid + 1
            else:
                start_compare = mid

            # Phase 2: Pop and compare with second half
            pop_steps = []
            for i in range(start_compare, n):
                # Compare with mirror position
                mirror_idx = n - 1 - i
                expected = symbols[mirror_idx]
                actual = symbols[i]

                if actual == expected:
                    pop_steps.append(f"pop={actual}")
                else:
                    pop_steps.append(f"pop≠{actual}")
                    # Early termination
                    all_steps = push_steps + pop_steps + ["F"]
                    return f"{input_string} # {' | '.join(all_steps)}"

            # Success case - add extra padding if configured
            trace_steps = push_steps + pop_steps
            natural_trace_length = len(trace_steps)

            # Compute extra padding
            extra_padding_count = compute_extra_padding_length(
                input_length=input_length,
                natural_trace_length=natural_trace_length,
                scale_type=padding_scale_type,
                multiplier=padding_multiplier,
                constant=padding_constant,
                max_length=padding_max,
            )

            if extra_padding_count > 0:
                # Natural trace + extra padding + final
                extra_padding_part = " ".join(["[PAD]"] * extra_padding_count)
                all_steps = trace_steps + [extra_padding_part, "T"]
                return f"{input_string} # {' | '.join(all_steps)}"
            else:
                # Just natural trace (backward compatible)
                all_steps = trace_steps + ["T"]
                return f"{input_string} # {' | '.join(all_steps)}"

    elif mode == "verify":
        # Show verification steps (more verbose than trace)
        symbols = input_string.strip().split()

        if marked:
            marker_idx = symbols.index("$") if "$" in symbols else -1
            if marker_idx == -1:
                return f"{input_string} # F"

            w = symbols[:marker_idx]
            w_reverse_actual = symbols[marker_idx + 1 :]

            # Build comparison trace
            steps = []
            w_reverse_expected = w[::-1]

            # Show comparison
            steps.append(f"reverse {' '.join(w_reverse_expected)}")
            steps.append(
                f"compare {' '.join(w_reverse_actual)} {' '.join(w_reverse_expected)}"
            )
            steps.append(result)

            return f"{input_string} # {' | '.join(steps)}"
        else:
            # For unmarked: compare forward and reverse
            reverse = symbols[::-1]
            steps = []
            steps.append(f"reverse {' '.join(reverse)}")
            steps.append(f"compare {' '.join(symbols)} {' '.join(reverse)}")
            steps.append(result)

            return f"{input_string} # {' | '.join(steps)}"

    else:
        raise ValueError(f"Unknown mode: {mode}")


def make_all_splits(
    marked: bool,
    alphabet: List[str],
    mode: str,
    seed: int,
    split_sizes: Dict[str, int],
    length_ranges: Dict[str, Tuple[int, int]],
    padding_scale_type: str = "natural",
    padding_multiplier: float = 0.0,
    padding_constant: Optional[int] = None,
    padding_max: Optional[int] = None,
    negative_ratio: float = 0.5,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Generates ALL splits (train/validation/test) with length-based stratification.

    Args:
        marked: If True, generates marked palindromes (w#w^R). If False, generates ww^R.
        alphabet: List of symbols to use (e.g., ['a', 'b'])
        mode: Format mode (trace/final_value/verify)
        seed: Random seed
        split_sizes: Dictionary with keys "train", "validation", "test" and values
                    as the number of examples needed for each split.
        length_ranges: Dictionary with keys "train", "validation", "test" and values
                      as (min_length, max_length) tuples for each split.
        padding_scale_type: Type of padding scaling ("natural", "linear", "quadratic", "cubic", "constant")
        padding_multiplier: Multiplier for extra padding
        padding_constant: Fixed amount of extra padding (for constant mode)
        padding_max: Maximum extra padding length
        negative_ratio: Ratio of negative examples (non-palindromes) to generate (default: 0.5 for 50/50 split)

    Returns:
        Dictionary with keys "train", "validation", "test" containing lists of examples.
    """
    random.seed(seed)

    # Generate each split independently with its own length ranges
    split_pools = {}

    for split_name in ["train", "validation", "test"]:
        min_len, max_len = length_ranges[split_name]
        num_examples = split_sizes[split_name]

        # Calculate how many positive and negative examples to generate
        num_negative = int(num_examples * negative_ratio)
        num_positive = num_examples - num_negative

        examples = []
        seen = set()  # Track unique examples
        attempts = 0
        max_attempts = num_examples * 1000

        # Generate positive examples (palindromes)
        while len([e for e in examples if e.get("label") == "positive"]) < num_positive and attempts < max_attempts:
            attempts += 1

            # Sample length uniformly from range
            length = random.randint(min_len, max_len)

            # Generate palindrome
            palindrome_str = generate_palindrome(length, alphabet, marked)

            # Check for duplicates
            if palindrome_str in seen:
                continue

            seen.add(palindrome_str)

            # Generate output based on mode
            text = get_palindrome_trace(
                palindrome_str,
                marked,
                mode,
                padding_scale_type=padding_scale_type,
                padding_multiplier=padding_multiplier,
                padding_constant=padding_constant,
                padding_max=padding_max,
            )

            examples.append({"text": text, "label": "positive"})

        # Generate negative examples (non-palindromes)
        attempts = 0
        while len([e for e in examples if e.get("label") == "negative"]) < num_negative and attempts < max_attempts:
            attempts += 1

            # Sample length uniformly from range
            length = random.randint(min_len, max_len)

            # Generate non-palindrome
            non_palindrome_str = generate_non_palindrome(length, alphabet, marked)

            # Check for duplicates
            if non_palindrome_str in seen:
                continue

            seen.add(non_palindrome_str)

            # Generate output based on mode (will correctly label as F)
            text = get_palindrome_trace(
                non_palindrome_str,
                marked,
                mode,
                padding_scale_type=padding_scale_type,
                padding_multiplier=padding_multiplier,
                padding_constant=padding_constant,
                padding_max=padding_max,
            )

            examples.append({"text": text, "label": "negative"})

        if len(examples) < num_examples:
            print(
                f"Warning: Could only generate {len(examples)}/{num_examples} examples for {split_name} "
                f"within length range [{min_len}, {max_len}] after {attempts} attempts."
            )

        random.shuffle(examples)
        split_pools[split_name] = examples

    return split_pools


def make_examples(
    num_examples: int,
    min_len: int,
    max_len: int,
    marked: bool,
    alphabet: List[str],
    mode: str,
    seed: int = None,
    padding_scale_type: str = "natural",
    padding_multiplier: float = 0.0,
    padding_constant: Optional[int] = None,
    padding_max: Optional[int] = None,
    negative_ratio: float = 0.5,
) -> List[Dict[str, str]]:
    """
    Generates palindrome examples (backward compatibility).

    Args:
        num_examples: Number of examples to generate
        min_len: Minimum palindrome length
        max_len: Maximum palindrome length
        marked: If True, generates marked palindromes
        alphabet: List of symbols to use
        mode: Format mode (trace/final_value/verify)
        seed: Random seed
        padding_scale_type: Type of padding scaling
        padding_multiplier: Multiplier for extra padding
        padding_constant: Fixed amount of extra padding
        padding_max: Maximum extra padding length
        negative_ratio: Ratio of negative examples (non-palindromes) to generate (default: 0.5 for 50/50 split)

    Returns:
        List of example dictionaries with "text" key
    """
    if seed is not None:
        random.seed(seed)

    # Calculate how many positive and negative examples to generate
    num_negative = int(num_examples * negative_ratio)
    num_positive = num_examples - num_negative

    examples = []
    seen = set()

    # Generate positive examples (palindromes)
    for _ in range(num_positive * 10):  # Allow retries for uniqueness
        if len([e for e in examples if e.get("label") == "positive"]) >= num_positive:
            break

        length = random.randint(min_len, max_len)
        palindrome_str = generate_palindrome(length, alphabet, marked)

        if palindrome_str in seen:
            continue

        seen.add(palindrome_str)
        text = get_palindrome_trace(
            palindrome_str,
            marked,
            mode,
            padding_scale_type=padding_scale_type,
            padding_multiplier=padding_multiplier,
            padding_constant=padding_constant,
            padding_max=padding_max,
        )
        examples.append({"text": text, "label": "positive"})

    # Generate negative examples (non-palindromes)
    for _ in range(num_negative * 10):  # Allow retries for uniqueness
        if len([e for e in examples if e.get("label") == "negative"]) >= num_negative:
            break

        length = random.randint(min_len, max_len)
        non_palindrome_str = generate_non_palindrome(length, alphabet, marked)

        if non_palindrome_str in seen:
            continue

        seen.add(non_palindrome_str)
        text = get_palindrome_trace(
            non_palindrome_str,
            marked,
            mode,
            padding_scale_type=padding_scale_type,
            padding_multiplier=padding_multiplier,
            padding_constant=padding_constant,
            padding_max=padding_max,
        )
        examples.append({"text": text, "label": "negative"})

    random.shuffle(examples)
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate palindrome examples over a given alphabet."
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=4,
        help="Minimum length of palindrome (number of symbols).",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=16,
        help="Maximum length of palindrome (number of symbols).",
    )
    parser.add_argument(
        "--marked",
        action="store_true",
        help="Generate marked palindromes (w#w^R). Default is unmarked (ww^R).",
    )
    parser.add_argument(
        "--alphabet",
        type=str,
        default="a,b",
        help="Comma-separated alphabet symbols (default: a,b).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="final_value",
        choices=["trace", "final_value", "verify"],
        help="Output format.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of examples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    alphabet = args.alphabet.split(",")

    print(
        f"Generating {args.num_examples} {'marked' if args.marked else 'unmarked'} palindromes"
    )
    print(f"Length range: [{args.min_len}, {args.max_len}]")
    print(f"Alphabet: {alphabet}")
    print(f"Format: {args.format}")

    examples = make_examples(
        num_examples=args.num_examples,
        min_len=args.min_len,
        max_len=args.max_len,
        marked=args.marked,
        alphabet=alphabet,
        mode=args.format,
        seed=args.seed,
    )

    print("\n--- Generated Examples ---")
    for i, ex in enumerate(examples):
        print(f"[{i+1}] {ex['text']}")
