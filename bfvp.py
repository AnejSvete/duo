import argparse
import logging
import random
from typing import Any, Dict, List, Set, Tuple, Optional

LOGGER = logging.getLogger(__name__)

# Added for structural consistency with the regular language codebase
BFVP_CREATORS = {
    "bfvp": True,
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
        extra = int((input_length ** 2) * multiplier)
    elif scale_type == "cubic":
        extra = int((input_length ** 3) * multiplier)
    else:
        raise ValueError(f"Unknown scale_type: {scale_type}")

    # Apply cap
    if max_length is not None:
        extra = min(extra, max_length)

    return max(0, extra)  # Never negative


def generate_formula_tree(depth: int, num_vars: int) -> Dict[str, Any]:
    """
    Generates an expression tree for a formula with fan-in equal to num_vars.
    """
    if num_vars <= 0:
        raise ValueError("num_vars must be positive.")
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    if num_vars < 2:
        raise ValueError("num_vars must be at least 2.")

    variables = [f"x{i}" for i in range(1, num_vars + 1)]

    def gen(current_depth: int) -> Dict[str, Any]:
        # Base case: we've reached the leaves.
        if current_depth <= 0:
            return {"var": random.choice(variables)}

        op = random.choice(["and", "or"])

        # Determine children based on depth using num_vars as fan-in.
        if current_depth == 1:
            # Nodes connected to leaves sample unique variables.
            leaf_vars = random.sample(variables, num_vars)
            children = [{"var": var} for var in leaf_vars]
        else:
            # Intermediate nodes recurse.
            children = [gen(current_depth - 1) for _ in range(num_vars)]

        # Randomly add negations to children.
        final_children = []
        for child in children:
            if random.random() < 0.25:
                final_children.append({"op": "not", "child": child})
            else:
                final_children.append(child)

        return {"op": op, "children": final_children}

    return gen(depth)


def substitute_vars_in_tree(
    node: Dict[str, Any], assignments: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Replaces all variable nodes in an expression tree with constant nodes ('T'/'F').
    """
    if "const" in node:
        return node
    if "var" in node:
        value = assignments.get(node["var"], False)  # Default to False
        return {"const": "T" if value else "F"}
    if node["op"] == "not":
        return {
            "op": "not",
            "child": substitute_vars_in_tree(node["child"], assignments),
        }
    else:  # 'and' or 'or'
        return {
            "op": node["op"],
            "children": [
                substitute_vars_in_tree(child, assignments)
                for child in node["children"]
            ],
        }


def get_variables_from_tree(node: Dict[str, Any]) -> Set[str]:
    """
    Traverses an expression tree and returns a set of unique variable names.
    """
    if "var" in node:
        return {node["var"]}
    if "const" in node:
        return set()
    if node["op"] == "not":
        return get_variables_from_tree(node["child"])

    # Union of variables from all children
    return set.union(*[get_variables_from_tree(child) for child in node["children"]])


def tree_to_prefix_str(tree: Dict[str, Any]) -> str:
    """Converts an expression tree to a prefix notation string (Polish Notation)."""
    if "const" in tree:
        return tree["const"]
    if "var" in tree:
        return tree["var"]

    op = tree["op"]
    if op == "not":
        child_str = tree_to_prefix_str(tree["child"])
        return f"{op} {child_str}"

    # 'and' or 'or'
    children_strs = [tree_to_prefix_str(child) for child in tree["children"]]
    return f"{op} {' '.join(children_strs)}"


def tree_to_postfix_str(tree: Dict[str, Any]) -> str:
    """Converts an expression tree to a postfix notation string (Reverse Polish Notation)."""
    if "const" in tree:
        return tree["const"]
    if "var" in tree:
        return tree["var"]

    op = tree["op"]
    if op == "not":
        child_str = tree_to_postfix_str(tree["child"])
        return f"{child_str} {op}"

    # 'and' or 'or'
    children_strs = [tree_to_postfix_str(child) for child in tree["children"]]
    return f"{' '.join(children_strs)} {op}"


def reduce_expression_tree_step(node: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Performs one layer of reduction on a variable-free expression tree.
    Returns the new tree and a boolean indicating if a reduction occurred.
    """
    was_reduced = False

    def reducer(n: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal was_reduced
        if "const" in n or "var" in n:
            return n

        if n["op"] == "not":
            reduced_child = reducer(n["child"])
            if reduced_child != n["child"]:
                return {"op": "not", "child": reduced_child}
        else:  # 'and' or 'or'
            reduced_children = [reducer(child) for child in n["children"]]
            if reduced_children != n["children"]:
                return {"op": n["op"], "children": reduced_children}

        # Check if the node is now reducible (i.e., all children are constants)
        is_reducible = (
            ("const" in n["child"])
            if n["op"] == "not"
            else all("const" in child for child in n.get("children", []))
        )

        if is_reducible:
            was_reduced = True
            if n["op"] == "not":
                result = not (n["child"]["const"] == "T")
            else:
                child_values = [child["const"] == "T" for child in n["children"]]
                if n["op"] == "and":
                    result = all(child_values)
                else:  # 'or'
                    result = any(child_values)
            return {"const": "T" if result else "F"}

        return n

    return reducer(node), was_reduced


def evaluate_expression_tree(start_tree: Dict[str, Any]) -> str:
    """
    Fully evaluates a variable-free expression tree to a single 'T' or 'F' constant.
    """
    current_tree = start_tree
    while "op" in current_tree:
        current_tree, reduced = reduce_expression_tree_step(current_tree)
        if not reduced and "op" in current_tree:
            raise ValueError("Expression tree could not be fully reduced.")
    return current_tree.get("const", "ERROR")


def get_postfix_reduction_steps(start_tree: Dict[str, Any]) -> List[str]:
    """
    Takes a variable-free expression tree and returns the evaluation trace
    as a list of postfix notation strings.
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
    Takes a variable-free expression tree and returns the full evaluation trace
    string, with each step in postfix notation.
    """
    steps = get_postfix_reduction_steps(start_tree)
    if len(steps) <= 1:
        return steps[0]
    return f"{steps[0]} # {' | '.join(steps[1:])}"


def make_all_splits(
    min_depth: int,
    max_depth: int,
    num_vars: int,
    mode: str,
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
        num_vars: Number of variables
        mode: Format mode (trace/final_value/empty_trace/lookup)
        seed: Random seed
        split_sizes: Dictionary with keys "train", "validation", "test" and values
                     as the number of examples needed for each split.
        depth_ranges: Optional dictionary with keys "train", "validation", "test" and values
                     as (min_depth, max_depth) tuples for each split. Used for generation sampling.
        length_ranges: Optional dictionary with keys "train", "validation", "test" and values
                     as (min_length, max_length) tuples for each split. Examples are filtered by final token length.
                     If provided, uses rejection sampling to ensure examples fall within length range.
        padding_scale_type: Type of padding scaling ("natural", "linear", "quadratic", "cubic", "constant")
        padding_multiplier: Multiplier for extra padding
        padding_constant: Fixed amount of extra padding (for constant mode)
        padding_max: Maximum extra padding length

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

    LOGGER.info(f"Starting BFVP data generation with seed={seed}")
    LOGGER.info(f"Split sizes: {split_sizes}")
    LOGGER.info(f"Depth ranges: {depth_ranges}")
    LOGGER.info(f"Length ranges: {length_ranges}")

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

            current_depth = random.randint(split_min_depth, split_max_depth)
            if current_depth not in depth_to_length_map:
                depth_to_length_map[current_depth] = []
            expression_tree = generate_formula_tree(current_depth, num_vars)
            variables = get_variables_from_tree(expression_tree)
            assignments = {var: random.choice([True, False]) for var in variables}
            substituted_tree = substitute_vars_in_tree(expression_tree, assignments)

            text = _generate_text_from_tree(
                substituted_tree,
                expression_tree,
                assignments,
                mode,
                padding_scale_type=padding_scale_type,
                padding_multiplier=padding_multiplier,
                padding_constant=padding_constant,
                padding_max=padding_max,
            )

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
                    depth_to_length_map[current_depth].append(text_length)

                    # Log progress periodically
                    if len(examples) % 1000 == 0:
                        LOGGER.info(f"  Progress: {len(examples)}/{num_examples} examples generated (attempts={attempts}, rejection_rate={rejected_count/attempts*100:.1f}%)")
                else:
                    rejected_count += 1
                    rejected_lengths.append(text_length)

                    # Log sample rejections to understand why examples are being rejected
                    if rejected_count <= 10 or (rejected_count % 1000 == 0):
                        LOGGER.debug(f"  Rejected example {rejected_count}: length={text_length} not in [{min_len}, {max_len}], depth={current_depth}")
            else:
                examples.append({"text": text})
                # Track lengths even when not filtering
                if "#" in text:
                    input_part = text.split("#")[0].strip()
                    text_length = len(input_part.split())
                else:
                    text_length = len(text.split())
                accepted_lengths.append(text_length)
                depth_to_length_map[current_depth].append(text_length)

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
    num_vars: int,
    mode: str,
    seed: int = None,
    padding_scale_type: str = "natural",
    padding_multiplier: float = 0.0,
    padding_constant: Optional[int] = None,
    padding_max: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Generates formulas based on the specified mode (backward compatibility).
    """
    if seed is not None:
        random.seed(seed)

    examples = []
    for _ in range(num_examples):
        current_depth = random.randint(min_depth, max_depth)
        expression_tree = generate_formula_tree(current_depth, num_vars)
        variables = get_variables_from_tree(expression_tree)
        assignments = {var: random.choice([True, False]) for var in variables}
        substituted_tree = substitute_vars_in_tree(expression_tree, assignments)

        text = _generate_text_from_tree(
            substituted_tree,
            expression_tree,
            assignments,
            mode,
            padding_scale_type=padding_scale_type,
            padding_multiplier=padding_multiplier,
            padding_constant=padding_constant,
            padding_max=padding_max,
        )
        examples.append({"text": text})

    return examples


def _generate_text_from_tree(
    substituted_tree: Dict[str, Any],
    expression_tree: Dict[str, Any],
    assignments: Dict[str, bool],
    mode: str,
    padding_scale_type: str = "natural",
    padding_multiplier: float = 0.0,
    padding_constant: Optional[int] = None,
    padding_max: Optional[int] = None,
) -> str:
    """Helper function to generate text representation from trees."""
    if mode == "trace":
        steps = get_postfix_reduction_steps(substituted_tree)
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
        postfix_str = tree_to_postfix_str(substituted_tree)
        final_value = evaluate_expression_tree(substituted_tree)
        text = f"{postfix_str} # {final_value}"

    elif mode == "empty_trace":
        steps = get_postfix_reduction_steps(substituted_tree)
        initial_repr = steps[0]
        input_length = len(initial_repr.split())

        if len(steps) > 1:
            reduction_steps_list = steps[1:]
            final_value = reduction_steps_list[-1]
            natural_trace_length = len(reduction_steps_list) - 1  # Exclude final

            # Create empty padding for natural trace structure - collect all padding
            total_pad_count = 0
            for step in reduction_steps_list[:-1]:
                num_tokens = len(step.split())
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

            # Combine all padding (natural + extra) and output without separators
            total_pad_count += extra_padding_count
            if total_pad_count > 0:
                all_padding = " ".join(["[PAD]"] * total_pad_count)
                text = f"{initial_repr} # {all_padding} {final_value}"
            else:
                text = f"{initial_repr} # {final_value}"
        else:
            text = initial_repr
    elif mode == "lookup":
        assignment_parts = []
        sorted_vars = sorted(list(assignments.keys()), key=lambda v: int(v[1:]))
        for var in sorted_vars:
            value = assignments[var]
            assignment_parts.append(f"{var} {'T' if value else 'F'}")
        assignment_str = " ".join(assignment_parts)
        initial_formula_str = tree_to_postfix_str(expression_tree)
        full_trace_str = get_postfix_reduction_trace(substituted_tree)
        trace_parts = full_trace_str.split(" # ", 1)
        if len(trace_parts) == 2:
            reduction_trace = trace_parts[1]
        else:
            reduction_trace = trace_parts[0]
        text = f"{assignment_str} | {initial_formula_str} # {reduction_trace}"
    else:
        raise ValueError(f"Unknown format mode: {mode}")

    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Boolean formulas in postfix notation."
    )
    parser.add_argument(
        "--min_depth",
        type=int,
        default=1,
        help="The minimum depth of the formula tree.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=3,
        help="The maximum depth of the formula tree.",
    )
    parser.add_argument(
        "--num_vars",
        type=int,
        default=2,
        help="Number of unique variables (also used as fan-in for all nodes).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="trace",
        choices=["trace", "final_value", "empty_trace", "lookup"],
        help="Output format: 'trace', 'final_value', 'empty_trace', or 'lookup'.",
    )
    parser.add_argument(
        "--num_examples", type=int, default=5, help="Number of examples to generate."
    )

    args = parser.parse_args()

    print(
        f"Generating {args.num_examples} examples with tree depth from {args.min_depth} to {args.max_depth}."
    )
    print(f"Number of variables (and fan-in): {args.num_vars}")
    print(f"Output format: '{args.format}'")

    examples = make_examples(
        num_examples=args.num_examples,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        num_vars=args.num_vars,
        mode=args.format,
    )

    print("\n--- Generated Examples ---")
    for i, ex in enumerate(examples):
        print(f"[{i+1}] {ex['text']}")
