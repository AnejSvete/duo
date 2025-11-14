import argparse
import random
from typing import Any, Dict, List, Set, Tuple

# Added for structural consistency with the regular language codebase
BFVP_CREATORS = {
    "bfvp": True,
}


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


def get_prefix_reduction_steps(start_tree: Dict[str, Any]) -> List[str]:
    """
    Takes a variable-free expression tree and returns the evaluation trace
    as a list of prefix notation strings.
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
    Takes a variable-free expression tree and returns the full evaluation trace
    string, with each step in prefix notation.
    """
    steps = get_prefix_reduction_steps(start_tree)
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

        examples = []
        attempts = 0
        max_attempts = num_examples * 1000  # Generous limit for rejection sampling

        while len(examples) < num_examples and attempts < max_attempts:
            attempts += 1

            current_depth = random.randint(split_min_depth, split_max_depth)
            expression_tree = generate_formula_tree(current_depth, num_vars)
            variables = get_variables_from_tree(expression_tree)
            assignments = {var: random.choice([True, False]) for var in variables}
            substituted_tree = substitute_vars_in_tree(expression_tree, assignments)

            text = _generate_text_from_tree(substituted_tree, expression_tree, assignments, mode)

            # Apply length filter if specified
            if use_length_filter:
                text_length = len(text.split())
                if min_len <= text_length <= max_len:
                    examples.append({"text": text})
            else:
                examples.append({"text": text})

        if len(examples) < num_examples:
            print(f"Warning: Could only generate {len(examples)}/{num_examples} examples for {split_name} "
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
            substituted_tree, expression_tree, assignments, mode
        )
        examples.append({"text": text})

    return examples


def _generate_text_from_tree(
    substituted_tree: Dict[str, Any],
    expression_tree: Dict[str, Any],
    assignments: Dict[str, bool],
    mode: str,
) -> str:
    """Helper function to generate text representation from trees."""
    if mode == "trace":
        text = get_prefix_reduction_trace(substituted_tree)
    elif mode == "final_value":
        prefix_str = tree_to_prefix_str(substituted_tree)
        final_value = evaluate_expression_tree(substituted_tree)
        text = f"{prefix_str} # {final_value}"
    elif mode == "empty_trace":
        steps = get_prefix_reduction_steps(substituted_tree)
        initial_repr = steps[0]
        if len(steps) > 1:
            reduction_steps_list = steps[1:]
            final_value = reduction_steps_list[-1]
            padded_steps = []
            for step in reduction_steps_list[:-1]:
                num_tokens = len(step.split())
                padded_steps.append(" ".join(["[PAD]"] * num_tokens))
            if padded_steps:
                padded_trace = " [PAD] ".join(padded_steps)
                text = f"{initial_repr} # {padded_trace} [PAD] {final_value}"
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
        initial_formula_str = tree_to_prefix_str(expression_tree)
        full_trace_str = get_prefix_reduction_trace(substituted_tree)
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
        description="Generate Boolean formulas in prefix notation."
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
