import argparse
import logging
import random
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    # Length is the total length of the final palindrome
    half_len = length // 2

    if marked:
        # For marked palindromes: generate w, then marker ($), then w^R (full reverse of w)
        w = [random.choice(alphabet) for _ in range(half_len)]
        w_reverse = w[::-1]
        # Optimized: pre-allocate list
        result = w + ["$"] + w_reverse
        return " ".join(result)
    else:
        # For unmarked palindromes: generate ww^R where the string reads same forwards/backwards
        first_half = [random.choice(alphabet) for _ in range(half_len)]

        # For even-length palindromes
        if length % 2 == 0:
            reverse_half = first_half[::-1]
            return " ".join(first_half + reverse_half)
        else:
            # For odd-length palindromes, add middle element
            middle = random.choice(alphabet)
            reverse_half = first_half[::-1]
            result = first_half + [middle] + reverse_half
            return " ".join(result)


def generate_non_palindrome(
    length: int, alphabet: List[str], marked: bool = True
) -> str:
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
        raise ValueError(
            "Alphabet must have at least 2 symbols to generate non-palindromes."
        )
    half_len = length // 2

    if marked:
        # Generate w and w', ensuring w' is NOT the reverse of w
        # More efficient: generate w, then create w' by copying reverse and changing one position
        w = [random.choice(alphabet) for _ in range(half_len)]
        w_reverse = w[::-1]
        w_prime = w_reverse.copy()  # Start with reverse

        # Force a difference at a random position
        diff_pos = random.randint(0, half_len - 1)
        current = w_prime[diff_pos]
        # Pre-filter alternatives for efficiency
        alternatives = [s for s in alphabet if s != current]
        if alternatives:
            w_prime[diff_pos] = random.choice(alternatives)
        else:
            # Edge case: alphabet size is 1, but this should be caught above
            raise RuntimeError(
                "Cannot generate non-palindrome with single-symbol alphabet"
            )

        return " ".join(w + ["$"] + w_prime)
    else:
        # Generate a string that is NOT a palindrome
        # More efficient: generate first half, then second half that differs
        first_half = [random.choice(alphabet) for _ in range(half_len)]

        if length % 2 == 0:
            # Even length: copy first half reversed, then change one position
            second_half = first_half[::-1].copy()
            if half_len > 0:
                diff_pos = random.randint(0, half_len - 1)
                current = second_half[diff_pos]
                alternatives = [s for s in alphabet if s != current]
                if alternatives:
                    second_half[diff_pos] = random.choice(alternatives)
                result = first_half + second_half
            else:
                # Edge case: length is 0
                result = []
        else:
            # Odd length: add middle, copy first half reversed, change one position
            middle = [random.choice(alphabet)]
            second_half = first_half[::-1].copy()
            if half_len > 0:
                diff_pos = random.randint(0, half_len - 1)
                current = second_half[diff_pos]
                alternatives = [s for s in alphabet if s != current]
                if alternatives:
                    second_half[diff_pos] = random.choice(alternatives)
            result = first_half + middle + second_half

        return " ".join(result)


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
        mode: Format mode (trace, final_value, empty_trace, or verify)
        padding_scale_type: Type of padding scaling ("natural", "linear", "quadratic", "cubic", "constant")
        padding_multiplier: Multiplier for extra padding
        padding_constant: Fixed amount of extra padding (for constant mode)
        padding_max: Maximum extra padding length

    Returns:
        Formatted trace string
    """
    is_palindrome = check_palindrome(input_string, marked)
    # print(f"Checking palindrome: '{input_string}' -> {'T' if is_palindrome else 'F'}")
    result = "T" if is_palindrome else "F"

    if mode == "final_value":
        # Just input and result (# separates prompt from completion)
        return f"{input_string} # {result}"

    elif mode == "empty_trace":
        # Generate trace structure but replace actual trace steps with [PAD] tokens
        symbols = input_string.strip().split()

        if marked:
            marker_idx = symbols.index("$") if "$" in symbols else -1
            if marker_idx == -1:
                return f"{input_string} # F"

            w = symbols[:marker_idx]
            w_reverse = symbols[marker_idx + 1 :]
            input_length = len(w) + len(w_reverse)

            # Generate trace steps (same logic as trace mode)
            push_steps = [f"push_{char}" for char in w]

            pop_steps = []
            stack = w.copy()
            mismatch = False

            for i, char in enumerate(w_reverse):
                if i < len(stack):
                    expected = stack[len(stack) - 1 - i]
                    if char == expected:
                        pop_steps.append(f"pop={char}")
                    else:
                        pop_steps.append(f"pop≠{char}")
                        mismatch = True
                        break
                else:
                    pop_steps.append(f"empty≠{char}")
                    mismatch = True
                    break

            if mismatch or len(w) != len(w_reverse):
                # For failed cases, still pad the trace - output without | separators
                trace_steps = push_steps + pop_steps
                total_pad_count = 0
                for step in trace_steps:
                    num_tokens = len(step.split("_"))  # Count tokens in step
                    total_pad_count += num_tokens
                if total_pad_count > 0:
                    all_padding = " ".join(["[PAD]"] * total_pad_count)
                    return f"{input_string} # {all_padding} F"
                else:
                    return f"{input_string} # F"
            else:
                # Success case - replace trace with padding
                trace_steps = push_steps + pop_steps
                natural_trace_length = len(trace_steps)

                # Replace each step with [PAD] tokens - collect all padding
                total_pad_count = 0
                for step in trace_steps:
                    num_tokens = len(step.split("_"))
                    total_pad_count += num_tokens

                # Compute extra padding
                extra_padding_count = compute_extra_padding_length(
                    input_length=input_length,
                    natural_trace_length=natural_trace_length,
                    scale_type=padding_scale_type,
                    multiplier=padding_multiplier,
                    constant=padding_constant,
                    max_length=padding_max,
                )

                # Combine all padding (natural + extra) and output without | separators
                total_pad_count += extra_padding_count
                if total_pad_count > 0:
                    all_padding = " ".join(["[PAD]"] * total_pad_count)
                    return f"{input_string} # {all_padding} T"
                else:
                    return f"{input_string} # T"

        else:
            # Unmarked palindromes
            n = len(symbols)
            mid = n // 2
            input_length = n

            push_steps = [f"push_{symbols[i]}" for i in range(mid)]

            if n % 2 == 1:
                push_steps.append(f"skip_{symbols[mid]}")
                start_compare = mid + 1
            else:
                start_compare = mid

            pop_steps = []
            mismatch = False
            for i in range(start_compare, n):
                mirror_idx = n - 1 - i
                expected = symbols[mirror_idx]
                actual = symbols[i]

                if actual == expected:
                    pop_steps.append(f"pop={actual}")
                else:
                    pop_steps.append(f"pop≠{actual}")
                    mismatch = True
                    break

            trace_steps = push_steps + pop_steps
            natural_trace_length = len(trace_steps)

            # Replace each step with [PAD] tokens - collect all padding
            total_pad_count = 0
            for step in trace_steps:
                num_tokens = len(step.split("_"))
                total_pad_count += num_tokens

            # Compute extra padding
            extra_padding_count = compute_extra_padding_length(
                input_length=input_length,
                natural_trace_length=natural_trace_length,
                scale_type=padding_scale_type,
                multiplier=padding_multiplier,
                constant=padding_constant,
                max_length=padding_max,
            )

            # Combine all padding (natural + extra) and output without | separators
            result_token = "T" if not mismatch else "F"
            total_pad_count += extra_padding_count
            if total_pad_count > 0:
                all_padding = " ".join(["[PAD]"] * total_pad_count)
                return f"{input_string} # {all_padding} {result_token}"
            else:
                return f"{input_string} # {result_token}"

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
                    # If trace_steps is empty, output padding without | separators
                    if not trace_steps:
                        return f"{input_string} # {extra_padding_part} T"
                    all_steps = trace_steps + [extra_padding_part, "T"]
                    return f"{input_string} # {' | '.join(all_steps)}"
                else:
                    # Just natural trace (backward compatible)
                    # If trace_steps is empty, just output the result
                    if not trace_steps:
                        return f"{input_string} # T"
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
                # If trace_steps is empty, output padding without | separators
                if not trace_steps:
                    return f"{input_string} # {extra_padding_part} T"
                all_steps = trace_steps + [extra_padding_part, "T"]
                return f"{input_string} # {' | '.join(all_steps)}"
            else:
                # Just natural trace (backward compatible)
                # If trace_steps is empty, just output the result
                if not trace_steps:
                    return f"{input_string} # T"
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

    logger.info("=" * 60)
    logger.info("Starting palindrome data generation")
    logger.info("=" * 60)
    logger.info(f"Type: {'Marked' if marked else 'Unmarked'} palindromes")
    logger.info(f"Alphabet: {alphabet} (size: {len(alphabet)})")
    logger.info(f"Mode: {mode}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Negative ratio: {negative_ratio:.1%}")
    if padding_multiplier > 0 or padding_constant:
        logger.info(
            f"Padding: scale_type={padding_scale_type}, multiplier={padding_multiplier}, constant={padding_constant}, max={padding_max}"
        )
    logger.info("")

    # Generate each split independently with its own length ranges
    split_pools = {}
    overall_start_time = time.time()

    for split_name in ["train", "validation", "test"]:
        split_start_time = time.time()
        min_len, max_len = length_ranges[split_name]
        num_examples = split_sizes[split_name]

        logger.info(f"Generating {split_name.upper()} split:")
        logger.info(f"  Target: {num_examples} examples")
        logger.info(f"  Length range: [{min_len}, {max_len}]")

        # Calculate how many positive and negative examples to generate
        num_negative = int(num_examples * negative_ratio)
        num_positive = num_examples - num_negative

        logger.info(f"  Positive (palindromes): {num_positive}")
        logger.info(f"  Negative (non-palindromes): {num_negative}")

        examples = []
        seen = set()  # Track unique examples
        max_attempts = num_examples * 1000

        # Track statistics
        duplicate_count = 0
        positive_count = 0
        negative_count = 0
        length_distribution = Counter()

        # Generate positive examples (palindromes)
        logger.info("  Generating positive examples...")
        attempts = 0
        last_log_time = time.time()

        while positive_count < num_positive and attempts < max_attempts:
            attempts += 1

            # Log progress every 2 seconds or every 1000 attempts
            current_time = time.time()
            if current_time - last_log_time > 2.0 or attempts % 1000 == 0:
                logger.info(
                    f"    Progress: {positive_count}/{num_positive} (attempts: {attempts}, duplicates: {duplicate_count})"
                )
                last_log_time = current_time

            # Sample length uniformly from range
            length = random.randint(min_len, max_len)

            # Generate palindrome
            palindrome_str = generate_palindrome(length, alphabet, marked)

            # Check for duplicates
            if palindrome_str in seen:
                duplicate_count += 1
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
            positive_count += 1
            length_distribution[length] += 1

        logger.info(f"    Completed: {positive_count}/{num_positive} positive examples")

        # Generate negative examples (non-palindromes)
        logger.info("  Generating negative examples...")
        attempts = 0
        last_log_time = time.time()
        neg_duplicate_count = 0

        while negative_count < num_negative and attempts < max_attempts:
            attempts += 1

            # Log progress every 2 seconds or every 1000 attempts
            current_time = time.time()
            if current_time - last_log_time > 2.0 or attempts % 1000 == 0:
                logger.info(
                    f"    Progress: {negative_count}/{num_negative} (attempts: {attempts}, duplicates: {neg_duplicate_count})"
                )
                last_log_time = current_time

            # Sample length uniformly from range
            length = random.randint(min_len, max_len)

            # Generate non-palindrome
            non_palindrome_str = generate_non_palindrome(length, alphabet, marked)

            # Check for duplicates
            if non_palindrome_str in seen:
                neg_duplicate_count += 1
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
            negative_count += 1
            length_distribution[length] += 1

        logger.info(f"    Completed: {negative_count}/{num_negative} negative examples")

        total_generated = positive_count + negative_count
        total_duplicates = duplicate_count + neg_duplicate_count

        if total_generated < num_examples:
            logger.warning(
                f"  WARNING: Only generated {total_generated}/{num_examples} examples "
                f"within length range [{min_len}, {max_len}]"
            )

        # Log statistics
        split_time = time.time() - split_start_time
        logger.info("  Statistics:")
        logger.info(f"    Total generated: {total_generated}/{num_examples}")
        logger.info(f"    Duplicates encountered: {total_duplicates}")
        logger.info(f"    Unique examples: {len(seen)}")
        logger.info(f"    Generation time: {split_time:.2f}s")
        logger.info(f"    Examples/second: {total_generated/split_time:.1f}")

        # Log length distribution
        if length_distribution:
            min_len_seen = min(length_distribution.keys())
            max_len_seen = max(length_distribution.keys())
            avg_len = (
                sum(length * count for length, count in length_distribution.items())
                / total_generated
            )
            logger.info(
                f"    Length stats: min={min_len_seen}, max={max_len_seen}, avg={avg_len:.1f}"
            )

            # Show distribution for small datasets or if highly skewed
            if total_generated <= 100 or len(length_distribution) <= 10:
                sorted_lengths = sorted(length_distribution.items())
                dist_str = ", ".join(
                    f"{length}:{count}" for length, count in sorted_lengths
                )
                logger.info(f"    Length distribution: {dist_str}")

        random.shuffle(examples)
        split_pools[split_name] = examples
        logger.info("")

    overall_time = time.time() - overall_start_time
    total_examples = sum(len(pool) for pool in split_pools.values())
    logger.info("=" * 60)
    logger.info("Generation complete!")
    logger.info(f"Total examples: {total_examples}")
    logger.info(f"Total time: {overall_time:.2f}s")
    logger.info(f"Overall rate: {total_examples/overall_time:.1f} examples/second")
    logger.info("=" * 60)

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
    start_time = time.time()

    if seed is not None:
        random.seed(seed)

    logger.info(
        f"Generating {num_examples} examples (length range: [{min_len}, {max_len}])"
    )

    # Calculate how many positive and negative examples to generate
    num_negative = int(num_examples * negative_ratio)
    num_positive = num_examples - num_negative

    logger.info(f"  Positive: {num_positive}, Negative: {num_negative}")

    examples = []
    seen = set()

    # Track counts efficiently
    positive_count = 0
    negative_count = 0
    duplicate_count = 0
    length_distribution = Counter()

    # Generate positive examples (palindromes)
    logger.info("  Generating positive examples...")
    attempts = 0
    max_attempts = num_positive * 10

    while positive_count < num_positive and attempts < max_attempts:
        attempts += 1
        length = random.randint(min_len, max_len)
        palindrome_str = generate_palindrome(length, alphabet, marked)

        if palindrome_str in seen:
            duplicate_count += 1
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
        positive_count += 1
        length_distribution[length] += 1

    logger.info(
        f"    Generated {positive_count}/{num_positive} (duplicates: {duplicate_count})"
    )

    # Generate negative examples (non-palindromes)
    logger.info("  Generating negative examples...")
    attempts = 0
    neg_duplicate_count = 0
    max_attempts = num_negative * 10

    while negative_count < num_negative and attempts < max_attempts:
        attempts += 1
        length = random.randint(min_len, max_len)
        non_palindrome_str = generate_non_palindrome(length, alphabet, marked)

        if non_palindrome_str in seen:
            neg_duplicate_count += 1
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
        negative_count += 1
        length_distribution[length] += 1

    logger.info(
        f"    Generated {negative_count}/{num_negative} (duplicates: {neg_duplicate_count})"
    )

    total_generated = positive_count + negative_count
    total_duplicates = duplicate_count + neg_duplicate_count
    elapsed_time = time.time() - start_time

    logger.info(
        f"  Total: {total_generated}/{num_examples} in {elapsed_time:.2f}s ({total_generated/elapsed_time:.1f} ex/s)"
    )
    logger.info(f"  Total duplicates: {total_duplicates}, Unique: {len(seen)}")

    if length_distribution:
        avg_len = (
            sum(length * count for length, count in length_distribution.items())
            / total_generated
        )
        logger.info(f"  Avg length: {avg_len:.1f}")

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
        choices=["trace", "final_value", "verify", "empty_trace"],
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
    parser.add_argument(
        "--negative_ratio",
        type=float,
        default=0.5,
        help="Ratio of negative examples (non-palindromes) to generate (default: 0.5).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (debug level).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all logging except errors.",
    )

    args = parser.parse_args()

    # Configure logging level based on arguments
    if args.quiet:
        logger.setLevel(logging.ERROR)
    elif args.verbose:
        logger.setLevel(logging.DEBUG)

    alphabet = args.alphabet.split(",")

    examples = make_examples(
        num_examples=args.num_examples,
        min_len=args.min_len,
        max_len=args.max_len,
        marked=args.marked,
        alphabet=alphabet,
        mode=args.format,
        seed=args.seed,
        negative_ratio=args.negative_ratio,
    )

    print("\n--- Generated Examples ---")
    for i, ex in enumerate(examples):
        label_marker = "✓" if ex["label"] == "positive" else "✗"
        print(f"[{i+1}] {label_marker} {ex['text']}")
