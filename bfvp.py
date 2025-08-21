import argparse
import random
from typing import Any, Dict, List, Set, Tuple


def generate_formula_tree(depth: int, num_vars: int, fan_in: int) -> Dict[str, Any]:
    """
    Generates an expression tree for a formula with a single, constant fan-in.
    """
    if num_vars <= 0:
        raise ValueError("num_vars must be positive.")
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    if fan_in < 2:
        raise ValueError("fan_in must be at least 2.")
    if fan_in > num_vars:
        raise ValueError(
            "fan_in cannot be greater than the number of unique variables."
        )

    variables = [f"x{i}" for i in range(1, num_vars + 1)]

    def gen(current_depth: int) -> Dict[str, Any]:
        # Base case: we've reached the leaves.
        if current_depth <= 0:
            return {"var": random.choice(variables)}

        op = random.choice(["and", "or"])

        # Determine children based on depth using the single fixed fan_in value.
        if current_depth == 1:
            # Nodes connected to leaves sample unique variables.
            leaf_vars = random.sample(variables, fan_in)
            children = [{"var": var} for var in leaf_vars]
        else:
            # Intermediate nodes recurse.
            children = [gen(current_depth - 1) for _ in range(fan_in)]

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


def get_prefix_reduction_trace(start_tree: Dict[str, Any]) -> str:
    """
    Takes a variable-free expression tree and returns the full evaluation trace
    string, with each step in prefix notation.
    """
    current_tree = start_tree
    steps = [tree_to_prefix_str(current_tree)]
    while "op" in current_tree:
        current_tree, reduced = reduce_expression_tree_step(current_tree)
        if not reduced:
            break
        steps.append(tree_to_prefix_str(current_tree))

    if len(steps) <= 1:
        return steps[0]
    return f"{steps[0]} # {' | '.join(steps[1:])}"


def make_examples(
    num_examples: int,
    max_depth: int,
    num_vars: int,
    fan_in: int,
    mode: str,
) -> List[Dict[str, str]]:
    """
    Generates formulas based on the specified mode.
    """
    examples = []
    for _ in range(num_examples):
        # For each example, choose a random depth between 1 and max_depth
        current_depth = random.randint(1, max_depth)
        expression_tree = generate_formula_tree(current_depth, num_vars, fan_in)

        variables = get_variables_from_tree(expression_tree)
        assignments = {var: random.choice([True, False]) for var in variables}
        substituted_tree = substitute_vars_in_tree(expression_tree, assignments)

        if mode == "trace":
            text = get_prefix_reduction_trace(substituted_tree)
        elif mode == "final_value":
            prefix_str = tree_to_prefix_str(substituted_tree)
            final_value = evaluate_expression_tree(substituted_tree)
            text = f"{prefix_str} # {final_value}"
        else:
            raise ValueError(f"Unknown format mode: {mode}")

        examples.append({"text": text})

    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Boolean formulas in prefix notation."
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
        default=5,
        help="Number of unique variables to choose from.",
    )
    parser.add_argument(
        "--fan_in",
        type=int,
        default=2,
        help="Fixed fan-in for all nodes.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="trace",
        choices=["trace", "final_value"],
        help="Output format: 'trace' for full reduction, 'final_value' for the result only.",
    )
    parser.add_argument(
        "--num_examples", type=int, default=5, help="Number of examples to generate."
    )
    parser.add_argument(
        "--show_examples", action="store_true", help="Print the generated examples."
    )

    args = parser.parse_args()

    print(
        f"Generating {args.num_examples} examples with max tree depth {args.max_depth}."
    )
    print(f"Variable pool size: {args.num_vars}")
    print(f"Fan-in: {args.fan_in}")
    print(f"Output format: '{args.format}'")

    examples = make_examples(
        num_examples=args.num_examples,
        max_depth=args.max_depth,
        num_vars=args.num_vars,
        fan_in=args.fan_in,
        mode=args.format,
    )

    if args.show_examples:
        print("\n--- Generated Examples ---")
        for i, ex in enumerate(examples):
            print(f"[{i+1}] {ex['text']}")
