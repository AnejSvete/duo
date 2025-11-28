import argparse
import logging
import random
from typing import Any, Dict, List, Optional, Set, Tuple

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
        extra = int((input_length**2) * multiplier)
    elif scale_type == "cubic":
        extra = int((input_length**3) * multiplier)
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
    Generates ALL splits (train/validation/test) using interdependent sampling.

    Each generated string is independently assigned to one of the three splits
    based on probabilistic sampling of remaining needs. This ensures no overlap
    between splits and mimics the FSA generation pattern.

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

    LOGGER.info(f"Starting BFVP data generation with seed={seed}")
    LOGGER.info(f"Split sizes: {split_sizes}")
    LOGGER.info(f"Depth ranges: {depth_ranges}")
    LOGGER.info(f"Length ranges: {length_ranges}")

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
        current_depth = random.randint(split_min_depth, split_max_depth)

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
    LOGGER.info(f"BFVP generation complete:")
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

            # Compute extra padding
            total_pad_count = compute_extra_padding_length(
                input_length=input_length,
                natural_trace_length=natural_trace_length,
                scale_type=padding_scale_type,
                multiplier=padding_multiplier,
                constant=padding_constant,
                max_length=padding_max,
            )

            # Combine all padding (natural + extra) and output without separators
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
