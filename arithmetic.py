import argparse
import logging
import random
from typing import Any, Dict, List, Set, Tuple, Optional

LOGGER = logging.getLogger(__name__)

# A creator dictionary, analogous to FSA_CREATORS and BFVP_CREATORS,
# for easy integration with the existing dataloader.
ARITHMETIC_CREATORS = {
    "arithmetic": True,
}

# --- Configuration Constants ---
OPERATORS = ["+", "-", "*", "/"]


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
        extra = int((input_length ** 2) * multiplier)
    elif scale_type == "cubic":
        extra = int((input_length ** 3) * multiplier)
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    # Apply cap
    if max_length is not None:
        extra = min(extra, max_length)

    return max(0, extra)  # Never negative


def generate_expression_tree(depth: int, min_val: int, max_val: int) -> Dict[str, Any]:
    """
    Generates a valid arithmetic expression tree using constraint-driven generation.

    This function ensures that all intermediate and final values of the expression
    remain within the [min_val, max_val] range.

    Args:
        depth: The desired depth of the expression tree.
        min_val: The minimum allowed value for any operand or result.
        max_val: The maximum allowed value for any operand or result.

    Returns:
        A dictionary representing the root of the expression tree.
    """
    # Base case: At depth 0, we are at a leaf, which must be a constant.
    if depth <= 0:
        return {"const": random.randint(min_val, max_val)}

    # Retry loop to handle cases where constraints might fail (e.g., finding a valid division).
    while True:
        op = random.choice(OPERATORS)
        try:
            if op == "+":
                # To get a + b = c, where c <= max_val:
                left_child = generate_expression_tree(depth - 1, min_val, max_val)
                left_val = evaluate_expression_tree(left_child)
                while True:
                    right_child = generate_expression_tree(depth - 1, min_val, max_val)
                    if evaluate_expression_tree(right_child) <= max_val - left_val:
                        break
                return {"op": op, "children": [left_child, right_child]}

            elif op == "-":
                # To get a - b = c, where c >= min_val:
                left_child = generate_expression_tree(depth - 1, min_val, max_val)
                left_val = evaluate_expression_tree(left_child)
                while True:
                    right_child = generate_expression_tree(depth - 1, min_val, max_val)
                    if evaluate_expression_tree(right_child) <= left_val - min_val:
                        break
                return {"op": op, "children": [left_child, right_child]}

            elif op == "*":
                # To get a * b = c, where c <= max_val:
                left_child = generate_expression_tree(depth - 1, min_val, max_val)
                left_val = evaluate_expression_tree(left_child)
                if left_val == 0:
                    right_child = generate_expression_tree(depth - 1, min_val, max_val)
                else:
                    while True:
                        right_child = generate_expression_tree(
                            depth - 1, min_val, max_val
                        )
                        if evaluate_expression_tree(right_child) <= max_val // left_val:
                            break
                return {"op": op, "children": [left_child, right_child]}

            elif op == "/":
                # Division is tricky. It's easier to work backward.
                while True:
                    right_child = generate_expression_tree(depth - 1, min_val, max_val)
                    right_val = evaluate_expression_tree(right_child)
                    if right_val != 0:
                        break
                result_child = generate_expression_tree(depth - 1, min_val, max_val)
                result_val = evaluate_expression_tree(result_child)
                left_val = right_val * result_val
                if not (min_val <= left_val <= max_val):
                    continue
                left_child = {"const": left_val}
                return {"op": op, "children": [left_child, right_child]}

        except (ValueError, ZeroDivisionError):
            continue


def evaluate_expression_tree(tree: Dict[str, Any]) -> int:
    """Recursively evaluates a valid expression tree to get its final integer value."""
    if "const" in tree:
        return tree["const"]

    child_values = [evaluate_expression_tree(child) for child in tree["children"]]
    op = tree["op"]

    if op == "+":
        return child_values[0] + child_values[1]
    if op == "-":
        return child_values[0] - child_values[1]
    if op == "*":
        return child_values[0] * child_values[1]
    if op == "/":
        return child_values[0] // child_values[1]
    raise ValueError(f"Unknown operator: {op}")


def tree_to_prefix_str(tree: Dict[str, Any]) -> str:
    """Converts an expression tree to a prefix notation string (Polish Notation)."""
    if "const" in tree:
        return str(tree["const"])
    if "var" in tree:
        return tree["var"]

    children_strs = [tree_to_prefix_str(child) for child in tree["children"]]
    op = tree["op"]

    return f"{op} {' '.join(children_strs)}"


def reduce_expression_tree_step(node: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Performs one reduction on the deepest, leftmost reducible sub-expression.
    """
    if "const" in node:
        return node, False

    new_children = []
    was_reduced = False
    for child in node["children"]:
        if not was_reduced:
            new_child, reduced_here = reduce_expression_tree_step(child)
            new_children.append(new_child)
            if reduced_here:
                was_reduced = True
        else:
            new_children.append(child)

    if was_reduced:
        return {"op": node["op"], "children": new_children}, True

    if all("const" in child for child in node["children"]):
        final_value = evaluate_expression_tree(node)
        return {"const": final_value}, True

    return node, False


def get_prefix_reduction_steps(start_tree: Dict[str, Any]) -> List[str]:
    """
    Generates the evaluation trace of an expression as a list of prefix strings.
    """
    current_tree = start_tree
    steps = [tree_to_prefix_str(current_tree)]
    while "op" in current_tree:
        current_tree, reduced = reduce_expression_tree_step(current_tree)
        if not reduced:
            break
        steps.append(tree_to_prefix_str(current_tree))
    return steps


def get_prefix_reduction_trace(start_tree: Dict[str, Any]) -> str:
    """
    Generates the full evaluation trace of an expression, with each step in prefix.
    """
    steps = get_prefix_reduction_steps(start_tree)
    if len(steps) <= 1:
        return steps[0]
    return f"{steps[0]} # {' | '.join(steps[1:])}"


def get_constants_from_tree(node: Dict[str, Any]) -> Set[int]:
    """Traverses an expression tree and returns a set of unique constant values."""
    if "const" in node:
        return {node["const"]}
    if "var" in node:
        return set()
    return set.union(
        *[get_constants_from_tree(child) for child in node.get("children", [])]
    )


def variablize_tree(
    node: Dict[str, Any], value_to_var_map: Dict[int, str]
) -> Dict[str, Any]:
    """
    Replaces constant nodes in a tree with variable nodes based on the provided mapping.
    """
    if "const" in node:
        value = node["const"]
        return {"var": value_to_var_map[value]} if value in value_to_var_map else node
    if "op" in node:
        return {
            "op": node["op"],
            "children": [
                variablize_tree(child, value_to_var_map) for child in node["children"]
            ],
        }
    return node


def make_all_splits(
    min_depth: int,
    max_depth: int,
    mode: str,
    min_val: int,
    max_val: int,
    seed: int,
    split_sizes: Dict[str, int],
    depth_ranges: Dict[str, Tuple[int, int]] = None,
    length_ranges: Dict[str, Tuple[int, int]] = None,
    padding_scale_type: str = "natural",
    padding_multiplier: float = 0.0,
    padding_constant: Optional[int] = None,
    padding_max: Optional[int] = None,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Generates ALL splits (train/validation/test) with length-based stratification.

    Args:
        min_depth: Default minimum depth (used if depth_ranges not provided)
        max_depth: Default maximum depth (used if depth_ranges not provided)
        mode: Format mode (trace/final_value/empty_trace/lookup)
        min_val: Minimum value for operands
        max_val: Maximum value for operands
        seed: Random seed
        split_sizes: Dictionary with keys "train", "validation", "test" and values
                     as the number of examples needed for each split.
        depth_ranges: Optional dictionary with keys "train", "validation", "test" and values
                     as (min_depth, max_depth) tuples for each split. Used for generation sampling.
        length_ranges: Optional dictionary with keys "train", "validation", "test" and values
                     as (min_length, max_length) tuples for each split. Examples are filtered by final token length.
        padding_scale_type: Type of padding scaling ("natural", "linear", "quadratic", "cubic", "constant")
        padding_multiplier: Multiplier for extra padding
        padding_constant: Fixed amount of extra padding (for constant mode)
        padding_max: Maximum extra padding length
                     If provided, uses rejection sampling to ensure examples fall within length range.

    Returns:
        Dictionary with keys "train", "validation", "test" containing lists of examples.
    """
    random.seed(seed)

    # If depth_ranges not provided, use same range for all splits
    if depth_ranges is None:
        depth_ranges = {
            "train": (min_depth, max_depth),
            "validation": (min_depth, max_depth),
            "test": (min_depth, max_depth),
        }

    # Generate each split independently with its own depth and length ranges
    split_pools = {}

    LOGGER.info(f"Starting arithmetic data generation with seed={seed}")
    LOGGER.info(f"Split sizes: {split_sizes}")
    LOGGER.info(f"Depth ranges: {depth_ranges}")
    LOGGER.info(f"Length ranges: {length_ranges}")
    LOGGER.info(f"Value range: [{min_val}, {max_val}]")

    for split_name in ["train", "validation", "test"]:
        split_min_depth, split_max_depth = depth_ranges[split_name]
        num_examples = split_sizes[split_name]

        # Get length range for this split if specified
        if length_ranges is not None and split_name in length_ranges:
            min_len, max_len = length_ranges[split_name]
            use_length_filter = True
        else:
            min_len, max_len = None, None
            use_length_filter = False

        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"Generating {split_name} split:")
        LOGGER.info(f"  Target: {num_examples} examples")
        LOGGER.info(f"  Depth range: [{split_min_depth}, {split_max_depth}]")
        if use_length_filter:
            LOGGER.info(f"  Length filter: [{min_len}, {max_len}] (applied to INPUT part before '#')")
        else:
            LOGGER.info(f"  Length filter: None (accepting all lengths)")
        LOGGER.info(f"{'='*80}")

        examples = []
        attempts = 0
        max_attempts = num_examples * 1000  # Generous limit for rejection sampling

        # Track statistics for debugging
        rejected_count = 0
        accepted_lengths = []
        rejected_lengths = []
        depth_to_length_map = {}  # Track depth -> lengths mapping

        while len(examples) < num_examples and attempts < max_attempts:
            attempts += 1

            depth = random.randint(split_min_depth, split_max_depth)
            if depth not in depth_to_length_map:
                depth_to_length_map[depth] = []
            expression_tree = generate_expression_tree(depth, min_val, max_val)

            text = _generate_arithmetic_text(
                expression_tree,
                mode,
                padding_scale_type=padding_scale_type,
                padding_multiplier=padding_multiplier,
                padding_constant=padding_constant,
                padding_max=padding_max,
            )
            if text is None:
                continue

            # Apply length filter if specified
            if use_length_filter:
                # Compute length based on INPUT part only (before '#')
                # This matches curriculum filtering logic and actual problem size
                if "#" in text:
                    input_part = text.split("#")[0].strip()
                    text_length = len(input_part.split())
                else:
                    text_length = len(text.split())

                if min_len <= text_length <= max_len:
                    examples.append({"text": text})
                    accepted_lengths.append(text_length)
                    depth_to_length_map[depth].append(text_length)

                    # Log progress periodically
                    if len(examples) % 1000 == 0:
                        LOGGER.info(f"  Progress: {len(examples)}/{num_examples} examples generated (attempts={attempts}, rejection_rate={rejected_count/attempts*100:.1f}%)")
                else:
                    rejected_count += 1
                    rejected_lengths.append(text_length)

                    # Log sample rejections to understand why examples are being rejected
                    if rejected_count <= 10 or (rejected_count % 1000 == 0):
                        LOGGER.debug(f"  Rejected example {rejected_count}: length={text_length} not in [{min_len}, {max_len}], depth={depth}")
            else:
                examples.append({"text": text})
                # Track lengths even when not filtering
                if "#" in text:
                    input_part = text.split("#")[0].strip()
                    text_length = len(input_part.split())
                else:
                    text_length = len(text.split())
                accepted_lengths.append(text_length)
                depth_to_length_map[depth].append(text_length)

                # Log progress periodically
                if len(examples) % 1000 == 0:
                    LOGGER.info(f"  Progress: {len(examples)}/{num_examples} examples generated (attempts={attempts})")

        # Print summary statistics for this split
        import numpy as np

        LOGGER.info(f"\n{'='*80}")
        LOGGER.info(f"{split_name.upper()} split generation complete:")
        LOGGER.info(f"  Generated: {len(examples)}/{num_examples} examples")
        LOGGER.info(f"  Total attempts: {attempts}")
        if use_length_filter:
            LOGGER.info(f"  Accepted: {len(accepted_lengths)}")
            LOGGER.info(f"  Rejected: {rejected_count}")
            LOGGER.info(f"  Rejection rate: {rejected_count/attempts*100:.1f}%")

        if accepted_lengths:
            accepted_array = np.array(accepted_lengths)
            LOGGER.info(f"\n  Accepted lengths distribution:")
            LOGGER.info(f"    Min: {accepted_array.min()}")
            LOGGER.info(f"    Max: {accepted_array.max()}")
            LOGGER.info(f"    Mean: {accepted_array.mean():.2f}")
            LOGGER.info(f"    Median: {np.median(accepted_array):.2f}")
            LOGGER.info(f"    Std: {np.std(accepted_array):.2f}")

            # Show percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            LOGGER.info(f"    Percentiles:")
            for p in percentiles:
                val = np.percentile(accepted_array, p)
                LOGGER.info(f"      {p}%: {val:.1f}")

            # Show histogram of lengths (binned)
            unique, counts = np.unique(accepted_array, return_counts=True)
            LOGGER.info(f"\n    Length histogram (top 20 most common):")
            sorted_idx = np.argsort(-counts)[:20]
            for idx in sorted_idx:
                length = unique[idx]
                count = counts[idx]
                percentage = count / len(accepted_array) * 100
                bar = '#' * int(percentage / 2)  # Simple bar chart
                LOGGER.info(f"      len={length:3d}: {count:5d} ({percentage:5.1f}%) {bar}")

        if rejected_lengths and use_length_filter:
            rejected_array = np.array(rejected_lengths)
            LOGGER.info(f"\n  Rejected lengths distribution:")
            LOGGER.info(f"    Min: {rejected_array.min()}")
            LOGGER.info(f"    Max: {rejected_array.max()}")
            LOGGER.info(f"    Mean: {rejected_array.mean():.2f}")
            LOGGER.info(f"    Median: {np.median(rejected_array):.2f}")

            # Show why examples were rejected
            too_short = np.sum(rejected_array < min_len)
            too_long = np.sum(rejected_array > max_len)
            LOGGER.info(f"    Too short (< {min_len}): {too_short} ({too_short/len(rejected_array)*100:.1f}%)")
            LOGGER.info(f"    Too long (> {max_len}): {too_long} ({too_long/len(rejected_array)*100:.1f}%)")

        # Show depth-to-length mapping
        if depth_to_length_map:
            LOGGER.info(f"\n  Depth-to-Length analysis:")
            for depth in sorted(depth_to_length_map.keys()):
                lengths = depth_to_length_map[depth]
                if lengths:
                    depth_array = np.array(lengths)
                    LOGGER.info(f"    Depth {depth}: n={len(lengths):5d}, "
                                f"len_range=[{depth_array.min():3d}, {depth_array.max():3d}], "
                                f"mean={depth_array.mean():6.2f}, median={np.median(depth_array):6.2f}")

        LOGGER.info(f"{'='*80}\n")

        if len(examples) < num_examples:
            LOGGER.warning(f"⚠️  Warning: Could only generate {len(examples)}/{num_examples} examples for {split_name} "
                  f"within length range [{min_len}, {max_len}] after {attempts} attempts. "
                  f"Consider widening the length range or depth range.")

        random.shuffle(examples)
        split_pools[split_name] = examples

    return split_pools


def make_examples(
    num_examples: int,
    min_depth: int,
    max_depth: int,
    mode: str,
    min_val: int,
    max_val: int,
    seed: int = None,
    padding_scale_type: str = "natural",
    padding_multiplier: float = 0.0,
    padding_constant: Optional[int] = None,
    padding_max: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Generates a list of arithmetic expression examples (backward compatibility).
    """
    if seed is not None:
        random.seed(seed)

    examples = []
    for _ in range(num_examples):
        depth = random.randint(min_depth, max_depth)
        expression_tree = generate_expression_tree(depth, min_val, max_val)

        text = _generate_arithmetic_text(
            expression_tree,
            mode,
            padding_scale_type=padding_scale_type,
            padding_multiplier=padding_multiplier,
            padding_constant=padding_constant,
            padding_max=padding_max,
        )
        if text is not None:
            examples.append({"text": text})

    return examples


def _generate_arithmetic_text(
    expression_tree: Dict[str, Any],
    mode: str,
    padding_scale_type: str = "natural",
    padding_multiplier: float = 0.0,
    padding_constant: Optional[int] = None,
    padding_max: Optional[int] = None,
) -> str:
    """Helper function to generate text representation from expression tree."""
    if mode == "trace":
        steps = get_prefix_reduction_steps(expression_tree)
        initial_repr = steps[0]
        input_length = len(initial_repr.split())

        if len(steps) > 1:
            trace_steps = steps[1:-1]  # Intermediate steps (exclude initial and final)
            final_value = steps[-1]
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

            # Build output
            if extra_padding_count > 0:
                # Natural trace + extra padding block + final
                extra_padding_part = " ".join(["[PAD]"] * extra_padding_count)
                # If trace_steps is empty, output padding without | separators
                if not trace_steps:
                    text = f"{initial_repr} # {extra_padding_part} {final_value}"
                else:
                    trace_part = " | ".join(trace_steps)
                    text = f"{initial_repr} # {trace_part} | {extra_padding_part} | {final_value}"
            else:
                # Just natural trace (current behavior)
                # If trace_steps is empty, just output the result
                if not trace_steps:
                    text = f"{initial_repr} # {final_value}"
                else:
                    trace_part = " | ".join(trace_steps)
                    text = f"{initial_repr} # {trace_part} | {final_value}"
        else:
            # No intermediate steps
            text = steps[0]

    elif mode == "final_value":
        prefix_str = tree_to_prefix_str(expression_tree)
        final_value = evaluate_expression_tree(expression_tree)
        text = f"{prefix_str} # {final_value}"

    elif mode == "empty_trace":
        steps = get_prefix_reduction_steps(expression_tree)
        initial_repr = steps[0]
        input_length = len(initial_repr.split())

        if len(steps) > 1:
            reduction_steps_list = steps[1:]
            final_value = reduction_steps_list[-1]
            natural_trace_length = len(reduction_steps_list) - 1  # Exclude final

            # Create empty padding for natural trace structure
            padded_steps = []
            for step in reduction_steps_list[:-1]:
                num_tokens = len(step.split())
                padded_steps.append(" ".join(["[PAD]"] * num_tokens))

            # Compute extra padding
            extra_padding_count = compute_extra_padding_length(
                input_length=input_length,
                natural_trace_length=natural_trace_length,
                scale_type=padding_scale_type,
                multiplier=padding_multiplier,
                constant=padding_constant,
                max_length=padding_max,
            )

            # Build output
            if padded_steps or extra_padding_count > 0:
                parts = []
                if padded_steps:
                    parts.append(" [PAD] ".join(padded_steps))  # Natural structure padding
                if extra_padding_count > 0:
                    extra_padding_part = " ".join(["[PAD]"] * extra_padding_count)
                    # If padded_steps is empty, don't use [PAD] as separator
                    if not padded_steps:
                        text = f"{initial_repr} # {extra_padding_part} {final_value}"
                    else:
                        parts.append(extra_padding_part)
                        parts.append(final_value)
                        text = f"{initial_repr} # {' [PAD] '.join(parts)}"
                elif padded_steps:
                    # Only padded_steps, no extra padding
                    parts.append(final_value)
                    text = f"{initial_repr} # {' [PAD] '.join(parts)}"
            else:
                text = f"{initial_repr} # {final_value}"
        else:
            text = initial_repr
    elif mode == "lookup":
        num_vars = 2  # Fixed to 2 variables for lookup mode

        unique_constants = sorted(list(get_constants_from_tree(expression_tree)))
        num_to_variablize = min(num_vars, len(unique_constants))

        if num_to_variablize == 0:
            # Fallback for simple trees with no variety in constants
            return get_prefix_reduction_trace(expression_tree)

        constants_to_variablize = random.sample(unique_constants, num_to_variablize)

        var_to_value_map = {
            f"x{i+1}": val for i, val in enumerate(constants_to_variablize)
        }
        value_to_var_map = {val: var for var, val in var_to_value_map.items()}

        variable_tree = variablize_tree(expression_tree, value_to_var_map)

        assignment_parts = []
        for var, val in sorted(
            var_to_value_map.items(), key=lambda item: int(item[0][1:])
        ):
            assignment_parts.append(f"{var} {val}")
        assignment_str = " ".join(assignment_parts)

        formula_with_vars_str = tree_to_prefix_str(variable_tree)

        full_trace_str = get_prefix_reduction_trace(expression_tree)
        trace_parts = full_trace_str.split(" # ", 1)
        reduction_trace = trace_parts[1] if len(trace_parts) == 2 else trace_parts[0]

        text = f"{assignment_str} | {formula_with_vars_str} # {reduction_trace}"
    else:
        raise ValueError(f"Unknown format mode: {mode}")

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate arithmetic expressions in prefix notation with bounded intermediate values.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_examples", type=int, default=10, help="Number of examples to generate."
    )
    parser.add_argument(
        "--min_depth",
        type=int,
        default=1,
        help="Minimum depth of the expression tree.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=4,
        help="Maximum depth of the expression tree.",
    )
    parser.add_argument(
        "--min_val",
        type=int,
        default=0,
        help="Minimum value for operands and results.",
    )
    parser.add_argument(
        "--max_val",
        type=int,
        default=50,
        help="Maximum value for operands and results.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="trace",
        choices=["trace", "final_value", "empty_trace", "lookup"],
        help="Output format. 'lookup' mode creates 2 variables.",
    )

    args = parser.parse_args()

    print(f"--- 🚀 Generating Arithmetic Expressions (Prefix Notation) 🚀 ---")
    print(f"Value Range: [{args.min_val}, {args.max_val}]")
    print(
        f"Generating {args.num_examples} examples with tree depth from {args.min_depth} to {args.max_depth}."
    )
    print(f"Output Format: '{args.format}'")
    if args.format == "lookup":
        print(f"Variables per example: up to 2")

    generated_examples = make_examples(
        num_examples=args.num_examples,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        mode=args.format,
        min_val=args.min_val,
        max_val=args.max_val,
    )

    print("\n--- Generated Examples ---")
    for i, ex in enumerate(generated_examples):
        print(f"[{i+1}] {ex['text']}")
    print("--------------------------\n--- ✅ Script complete ---")
