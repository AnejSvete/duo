import argparse
import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

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
        extra = int((input_length**2) * multiplier)
    elif scale_type == "cubic":
        extra = int((input_length**3) * multiplier)
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


def tree_to_postfix_str(tree: Dict[str, Any]) -> str:
    """Converts an expression tree to a postfix notation string (Reverse Polish Notation)."""
    if "const" in tree:
        return str(tree["const"])
    if "var" in tree:
        return tree["var"]

    children_strs = [tree_to_postfix_str(child) for child in tree["children"]]
    op = tree["op"]

    return f"{' '.join(children_strs)} {op}"


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


def get_postfix_reduction_steps(start_tree: Dict[str, Any]) -> List[str]:
    """
    Generates the evaluation trace of an expression as a list of postfix strings.
    """
    current_tree = start_tree
    steps = [tree_to_postfix_str(current_tree)]
    while "op" in current_tree:
        current_tree, reduced = reduce_expression_tree_step(current_tree)
        if not reduced:
            break
        steps.append(tree_to_postfix_str(current_tree))
    return steps


def get_postfix_reduction_trace(start_tree: Dict[str, Any]) -> str:
    """
    Generates the full evaluation trace of an expression, with each step in postfix.
    """
    steps = get_postfix_reduction_steps(start_tree)
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
    Generates ALL splits (train/validation/test) using interdependent sampling.

    Each generated string is independently assigned to one of the three splits
    based on probabilistic sampling of remaining needs. This ensures no overlap
    between splits and mimics the FSA generation pattern.

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

    LOGGER.info(f"Starting arithmetic data generation with seed={seed}")
    LOGGER.info(f"Split sizes: {split_sizes}")
    LOGGER.info(f"Depth ranges: {depth_ranges}")
    LOGGER.info(f"Length ranges: {length_ranges}")
    LOGGER.info(f"Value range: [{min_val}, {max_val}]")

    # Interdependent sampling: generate examples and assign to splits dynamically
    assignment = {}  # Maps text -> assigned split
    counts = {s: 0 for s in ["train", "validation", "test"]}
    split_pools = {s: [] for s in ["train", "validation", "test"]}

    total_needed = sum(split_sizes.values())
    max_attempts = total_needed * 100
    attempts = 0

    while sum(counts.values()) < total_needed and attempts < max_attempts:
        attempts += 1

        # First decide which split to target based on remaining needs
        remaining = {
            s: max(0, split_sizes[s] - counts[s])
            for s in ["train", "validation", "test"]
        }
        total_remaining = sum(remaining.values())

        if total_remaining == 0:
            break

        # Probabilistically select a target split based on remaining needs
        p = random.random()
        cumulative = 0.0
        target_split = None
        for s in ["train", "validation", "test"]:
            cumulative += remaining[s] / total_remaining
            if p < cumulative:
                target_split = s
                break

        if target_split is None:
            target_split = "test"  # Fallback

        # Generate a candidate example using the target split's depth and length range
        split_min_depth, split_max_depth = depth_ranges[target_split]
        depth = random.randint(split_min_depth, split_max_depth)

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

        # Check if we've seen this example before
        if text in assignment:
            # Already assigned to a split - skip to avoid cross-contamination
            continue

        # Apply length filter if specified for the target split
        if length_ranges is not None and target_split in length_ranges:
            min_len, max_len = length_ranges[target_split]
            # Compute length based on INPUT part only (before '#')
            if "#" in text:
                input_part = text.split("#")[0].strip()
                text_length = len(input_part.split())
            else:
                text_length = len(text.split())

            if not (min_len <= text_length <= max_len):
                # Example doesn't match target split's length range, skip it
                continue

        # New example that passes all filters - assign it to the target split
        assignment[text] = target_split
        split_pools[target_split].append({"text": text})
        counts[target_split] += 1

        # Log progress periodically
        if sum(counts.values()) % 1000 == 0:
            LOGGER.info(f"Progress: {counts} / {split_sizes} (attempts={attempts})")

    # Shuffle each pool
    for pool in split_pools.values():
        random.shuffle(pool)

    # Log final statistics
    import numpy as np

    LOGGER.info(f"\n{'='*80}")
    LOGGER.info(f"Arithmetic generation complete:")
    LOGGER.info(f"  Total attempts: {attempts}")
    LOGGER.info(f"  Total unique examples: {len(assignment)}")

    for split_name in ["train", "validation", "test"]:
        examples = split_pools[split_name]
        LOGGER.info(f"\n{split_name.upper()} split:")
        LOGGER.info(f"  Generated: {len(examples)}/{split_sizes[split_name]} examples")

        if len(examples) < split_sizes[split_name]:
            LOGGER.warning(
                f"⚠️  Warning: Could only generate {len(examples)}/{split_sizes[split_name]} examples for {split_name}"
            )

        # Compute length statistics
        if examples:
            lengths = []
            for ex in examples:
                text = ex["text"]
                if "#" in text:
                    input_part = text.split("#")[0].strip()
                    text_length = len(input_part.split())
                else:
                    text_length = len(text.split())
                lengths.append(text_length)

            lengths_array = np.array(lengths)
            LOGGER.info(
                f"  Length stats: min={lengths_array.min()}, max={lengths_array.max()}, "
                f"mean={lengths_array.mean():.2f}, median={np.median(lengths_array):.2f}"
            )

    LOGGER.info(f"{'='*80}\n")

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
        steps = get_postfix_reduction_steps(expression_tree)
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
        postfix_str = tree_to_postfix_str(expression_tree)
        final_value = evaluate_expression_tree(expression_tree)
        text = f"{postfix_str} # {final_value}"

    elif mode == "empty_trace":
        steps = get_postfix_reduction_steps(expression_tree)
        initial_repr = steps[0]
        input_length = len(initial_repr.split())

        if len(steps) > 1:
            reduction_steps_list = steps[1:]
            final_value = reduction_steps_list[-1]
            natural_trace_length = len(reduction_steps_list) - 1  # Exclude final

            # Compute extra padding
            total_pad_count = compute_extra_padding_length(
                input_length=input_length,
                natural_trace_length=natural_trace_length,
                scale_type=padding_scale_type,
                multiplier=padding_multiplier,
                constant=padding_constant,
                max_length=padding_max,
            )

            if total_pad_count > 0:
                all_padding = " ".join(["[PAD]"] * total_pad_count)
                text = f"{initial_repr} # {all_padding} {final_value}"
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
            return get_postfix_reduction_trace(expression_tree)

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

        formula_with_vars_str = tree_to_postfix_str(variable_tree)

        full_trace_str = get_postfix_reduction_trace(expression_tree)
        trace_parts = full_trace_str.split(" # ", 1)
        reduction_trace = trace_parts[1] if len(trace_parts) == 2 else trace_parts[0]

        text = f"{assignment_str} | {formula_with_vars_str} # {reduction_trace}"
    else:
        raise ValueError(f"Unknown format mode: {mode}")

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate arithmetic expressions in postfix notation with bounded intermediate values.",
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

    print(f"--- 🚀 Generating Arithmetic Expressions (Postfix Notation) 🚀 ---")
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
