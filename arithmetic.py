import argparse
import random
from typing import Any, Dict, List, Set, Tuple

# A creator dictionary, analogous to FSA_CREATORS and BFVP_CREATORS,
# for easy integration with the existing dataloader.
ARITHMETIC_CREATORS = {
    "arithmetic": True,
}

# --- Configuration Constants ---
OPERATORS = ["+", "-", "*", "/"]


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


def make_examples(
    num_examples: int,
    min_depth: int,
    max_depth: int,
    mode: str,
    min_val: int,
    max_val: int,
    num_vars: int,
) -> List[Dict[str, str]]:
    """
    Generates a list of arithmetic expression examples.
    """
    examples = []
    for _ in range(num_examples):
        depth = random.randint(min_depth, max_depth)
        expression_tree = generate_expression_tree(depth, min_val, max_val)

        if mode == "trace":
            text = get_prefix_reduction_trace(expression_tree)
        elif mode == "final_value":
            prefix_str = tree_to_prefix_str(expression_tree)
            final_value = evaluate_expression_tree(expression_tree)
            text = f"{prefix_str} # {final_value}"
        elif mode == "empty_trace":
            steps = get_prefix_reduction_steps(expression_tree)
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
            if num_vars <= 0:
                raise ValueError("num_vars must be positive for 'lookup' mode.")

            unique_constants = sorted(list(get_constants_from_tree(expression_tree)))
            num_to_variablize = min(num_vars, len(unique_constants))

            if num_to_variablize == 0:
                # Fallback for simple trees with no variety in constants
                text = get_prefix_reduction_trace(expression_tree)
                examples.append({"text": text})
                continue

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
            reduction_trace = (
                trace_parts[1] if len(trace_parts) == 2 else trace_parts[0]
            )

            text = f"{assignment_str} | {formula_with_vars_str} # {reduction_trace}"
        else:
            raise ValueError(f"Unknown format mode: {mode}")

        examples.append({"text": text})

    return examples


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
        "--num_vars",
        type=int,
        default=2,
        help="Number of unique variables to create for 'lookup' mode.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="trace",
        choices=["trace", "final_value", "empty_trace", "lookup"],
        help="Output format. 'lookup' mode creates variables.",
    )

    args = parser.parse_args()

    print(f"--- 🚀 Generating Arithmetic Expressions (Prefix Notation) 🚀 ---")
    print(f"Value Range: [{args.min_val}, {args.max_val}]")
    print(
        f"Generating {args.num_examples} examples with tree depth from {args.min_depth} to {args.max_depth}."
    )
    print(f"Output Format: '{args.format}'")
    if args.format == "lookup":
        print(f"Variables per example: up to {args.num_vars}")

    generated_examples = make_examples(
        num_examples=args.num_examples,
        min_depth=args.min_depth,
        max_depth=args.max_depth,
        mode=args.format,
        min_val=args.min_val,
        max_val=args.max_val,
        num_vars=args.num_vars,
    )

    print("\n--- Generated Examples ---")
    for i, ex in enumerate(generated_examples):
        print(f"[{i+1}] {ex['text']}")
    print("--------------------------\n--- ✅ Script complete ---")
