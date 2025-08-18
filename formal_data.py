import random
from typing import Any, Dict, List, Set, Tuple


def generate_nc1_formula_tree(depth: int, num_vars: int) -> Dict[str, Any]:
    """
    Generates an expression tree for an NC1 formula by creating a full,
    balanced binary tree of a specified depth. The length of the formula
    (number of leaves) will be 2**depth.
    """
    if num_vars <= 0:
        raise ValueError("num_vars must be positive.")
    if depth < 0:
        raise ValueError("depth must be non-negative.")

    variables = [f"x{i}" for i in range(1, num_vars + 1)]

    def gen(current_depth: int) -> Dict[str, Any]:
        # Base case: we've reached the desired depth, create a leaf node.
        if current_depth <= 0:
            return {"var": random.choice(variables)}

        # Recursive step: create a binary operation node.
        op = random.choice(["and", "or"])
        left_child = gen(current_depth - 1)
        right_child = gen(current_depth - 1)

        # Randomly add negations to children
        if random.random() < 0.25:
            left_child = {"op": "not", "child": left_child}
        if random.random() < 0.25:
            right_child = {"op": "not", "child": right_child}

        return {"op": op, "left": left_child, "right": right_child}

    # Handle depth 0 case specifically, which is just a single variable.
    if depth == 0:
        return {"var": random.choice(variables)}

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
            "left": substitute_vars_in_tree(node["left"], assignments),
            "right": substitute_vars_in_tree(node["right"], assignments),
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
    return get_variables_from_tree(node["left"]) | get_variables_from_tree(
        node["right"]
    )


def expression_tree_to_str(tree: Dict[str, Any]) -> str:
    """
    Converts an expression tree to its parenthesized string representation.
    """
    if "const" in tree:
        return tree["const"]
    if "var" in tree:
        return tree["var"]
    if tree["op"] == "not":
        return f"not ( {expression_tree_to_str(tree['child'])} )"
    return f"( {expression_tree_to_str(tree['left'])} {tree['op']} {expression_tree_to_str(tree['right'])} )"


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
        else:
            reduced_left = reducer(n["left"])
            reduced_right = reducer(n["right"])
            if reduced_left != n["left"] or reduced_right != n["right"]:
                return {"op": n["op"], "left": reduced_left, "right": reduced_right}

        is_reducible = (
            ("const" in n["child"])
            if n["op"] == "not"
            else ("const" in n.get("left", {}) and "const" in n.get("right", {}))
        )
        if is_reducible:
            was_reduced = True
            if n["op"] == "not":
                result = not (n["child"]["const"] == "T")
            else:
                left_val, right_val = (
                    n["left"]["const"] == "T",
                    n["right"]["const"] == "T",
                )
                result = (
                    (left_val and right_val)
                    if n["op"] == "and"
                    else (left_val or right_val)
                )
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


def get_reduction_trace(start_tree: Dict[str, Any]) -> str:
    """
    Takes a variable-free expression tree and returns the full evaluation trace string.
    """
    current_tree = start_tree
    steps = [expression_tree_to_str(current_tree)]
    while "op" in current_tree:
        current_tree, reduced = reduce_expression_tree_step(current_tree)
        if not reduced:
            break
        steps.append(expression_tree_to_str(current_tree))

    if len(steps) <= 1:
        return steps[0]
    return f"{steps[0]} # {' | '.join(steps[1:])}"


def make_nc1_examples(
    num_examples: int, max_depth: int, num_vars: int, mode: str
) -> List[Dict[str, str]]:
    """
    Generates NC1 formulas based on the specified mode.

    Args:
        num_examples: The number of examples to generate.
        max_depth: The maximum depth of the formula's expression tree.
        num_vars: The number of unique variables available.
        mode: The output format. One of 'full_trace' or 'final_value'.
    """
    examples = []
    for _ in range(num_examples):
        # For each example, choose a random depth between 1 and max_depth
        current_depth = random.randint(1, max_depth)
        expression_tree = generate_nc1_formula_tree(current_depth, num_vars)

        variables = get_variables_from_tree(expression_tree)
        assignments = {var: random.choice([True, False]) for var in variables}
        substituted_tree = substitute_vars_in_tree(expression_tree, assignments)

        if mode == "full_trace":
            trace = get_reduction_trace(substituted_tree)
            examples.append({"text": trace})
        elif mode == "final_value":
            substituted_formula_str = expression_tree_to_str(substituted_tree)
            final_value = evaluate_expression_tree(substituted_tree)
            text = f"{substituted_formula_str} # {final_value}"
            examples.append({"text": text})

    return examples


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate NC1 Boolean formulas.")
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
        help="Number of unique variables to choose from.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="full_trace",
        choices=["full_trace", "final_value"],
        help="The output format for the generated examples.",
    )
    parser.add_argument(
        "--num_examples", type=int, default=5, help="Number of examples to generate."
    )
    parser.add_argument(
        "--show_examples", action="store_true", help="Print the generated examples."
    )

    args = parser.parse_args()

    print(
        f"Generating {args.num_examples} NC1 examples with max tree depth {args.max_depth}."
    )
    print(f"Variable pool size: {args.num_vars}")
    print(f"Output format: '{args.format}'")

    examples = make_nc1_examples(
        num_examples=args.num_examples,
        max_depth=args.max_depth,
        num_vars=args.num_vars,
        mode=args.format,
    )

    if args.show_examples:
        print("\n--- Generated Examples ---")
        for i, ex in enumerate(examples):
            print(f"[{i+1}] {ex['text']}")
