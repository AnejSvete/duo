import math
import random
from typing import Any, Dict, Set, Tuple

import datasets

# --- AST Generation ---


def generate_nc1_formula_ast(num_vars: int) -> Dict[str, Any]:
    """
    Generates an AST for a formula likely to be NC1-complete by creating
    a balanced tree with logarithmic depth based on the number of variables.
    """
    if num_vars <= 0:
        return {"const": random.choice(["T", "F"])}
    target_depth = math.ceil(math.log2(num_vars))
    variables = [f"x{i}" for i in range(1, num_vars + 1)]

    def gen(depth: int) -> Dict[str, Any]:
        if depth <= 0:
            return {"var": random.choice(variables)}
        op = random.choice(["and", "or"])
        left_child, right_child = gen(depth - 1), gen(depth - 1)
        if random.random() < 0.25:
            left_child = {"op": "not", "child": left_child}
        if random.random() < 0.25:
            right_child = {"op": "not", "child": right_child}
        return {"op": op, "left": left_child, "right": right_child}

    return gen(target_depth)


def random_bfvp_ast(max_depth=3, max_ops=6) -> Dict[str, Any]:
    """
    Generates a random variable-free formula AST.
    """
    ops, constants = ["and", "or"], ["T", "F"]

    def gen(depth, ops_left):
        if (
            depth <= 0
            or ops_left <= 0
            or (depth < max_depth and random.random() < 0.25)
        ):
            return {"const": random.choice(constants)}
        op = random.choice(ops + ["not"])
        if ops_left == 1 and op != "not":
            op = "not"
        if op == "not":
            return {"op": "not", "child": gen(depth - 1, ops_left - 1)}
        else:
            remaining_ops = ops_left - 1
            ops_for_left = random.randint(0, remaining_ops)
            ops_for_right = remaining_ops - ops_for_left
            return {
                "op": op,
                "left": gen(depth - 1, ops_for_left),
                "right": gen(depth - 1, ops_for_right),
            }

    return gen(max_depth, max_ops)


# --- AST Traversal and Evaluation ---


def substitute_vars_in_ast(
    node: Dict[str, Any], assignments: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Replaces all variable nodes in an AST with constant nodes based on an assignment map.
    """
    if "const" in node:
        return node
    if "var" in node:
        value = assignments.get(node["var"], False)
        return {"const": "T" if value else "F"}
    if node["op"] == "not":
        return {
            "op": "not",
            "child": substitute_vars_in_ast(node["child"], assignments),
        }
    else:
        return {
            "op": node["op"],
            "left": substitute_vars_in_ast(node["left"], assignments),
            "right": substitute_vars_in_ast(node["right"], assignments),
        }


def get_variables_from_ast(node: Dict[str, Any]) -> Set[str]:
    """
    Traverses an AST and returns a set of unique variable names.
    """
    if "var" in node:
        return {node["var"]}
    if "const" in node:
        return set()
    if node["op"] == "not":
        return get_variables_from_ast(node["child"])
    return get_variables_from_ast(node["left"]) | get_variables_from_ast(node["right"])


def ast_to_str(ast: Dict[str, Any]) -> str:
    """
    Converts an AST to its string representation.
    """
    if "const" in ast:
        return ast["const"]
    if "var" in ast:
        return ast["var"]
    if ast["op"] == "not":
        return f"not ( {ast_to_str(ast['child'])} )"
    return f"( {ast_to_str(ast['left'])} {ast['op']} {ast_to_str(ast['right'])} )"


def reduce_ast_step(node: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    was_reduced = False

    def reducer(n: Dict[str, Any]) -> Dict[str, Any]:
        nonlocal was_reduced
        if "const" in n:
            return n
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
        else:
            return (
                {"op": n["op"], "child": reducer(n["child"])}
                if n["op"] == "not"
                else {
                    "op": n["op"],
                    "left": reducer(n["left"]),
                    "right": reducer(n["right"]),
                }
            )

    return reducer(node), was_reduced


# --- Example Formatting ---


def get_reduction_trace(start_ast: Dict[str, Any]) -> str:
    """
    Takes a variable-free AST and returns the evaluation trace string.
    """
    current_ast = start_ast
    steps = [ast_to_str(current_ast)]
    while "op" in current_ast:
        current_ast, reduced = reduce_ast_step(current_ast)
        if not reduced:
            break
        steps.append(ast_to_str(current_ast))

    if len(steps) <= 1:
        return steps[0]

    # Separate the initial formula from the evaluation steps with a '#'
    return f"{steps[0]} # {' | '.join(steps[1:])}"


def make_nc1_examples(
    num_examples: int, num_vars: int, evaluate: bool
) -> list[Dict[str, str]]:
    """
    Generates NC1 formulas. If evaluate, the 'text' is the reduction trace.
    """
    examples = []
    for _ in range(num_examples):
        ast_with_vars = generate_nc1_formula_ast(num_vars)

        if not evaluate:
            examples.append({"text": ast_to_str(ast_with_vars)})
        else:
            variables = get_variables_from_ast(ast_with_vars)
            assignments = {var: random.choice([True, False]) for var in variables}
            substituted_ast = substitute_vars_in_ast(ast_with_vars, assignments)
            trace = get_reduction_trace(substituted_ast)
            examples.append({"text": trace})

    return examples


def make_bfvp_examples(
    num_examples=1000, max_depth=3, max_ops=6
) -> list[Dict[str, str]]:
    """
    Generates and evaluates variable-free formulas.
    """
    examples = []
    for _ in range(num_examples):
        ast = random_bfvp_ast(max_depth=max_depth, max_ops=max_ops)
        trace = get_reduction_trace(ast)
        examples.append({"text": trace})
    return examples


# --- Main Execution Block (for self-testing) ---
if __name__ == "__main__":
    import argparse
    import pprint

    parser = argparse.ArgumentParser(description="Generate Boolean formulas.")
    parser.add_argument(
        "--mode",
        type=str,
        default="bfvp",
        choices=["bfvp", "nc1"],
        help="Generation mode.",
    )
    parser.add_argument(
        "--num_examples", type=int, default=2, help="Number of examples to generate."
    )
    parser.add_argument(
        "--show_examples", action="store_true", help="Print the generated examples."
    )

    parser.add_argument(
        "--num_vars", type=int, default=4, help="Number of variables for NC1 mode."
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="For NC1 mode, assign values and show full evaluation trace.",
    )

    parser.add_argument(
        "--max_depth", type=int, default=4, help="Maximum formula depth for BFVP mode."
    )
    parser.add_argument(
        "--max_ops", type=int, default=8, help="Maximum operations for BFVP mode."
    )
    args = parser.parse_args()

    print(f"Running in '{args.mode}' mode.")

    if args.mode == "nc1":
        if args.evaluate:
            print(
                f"Generating and evaluating {args.num_examples} formula(s) with approx. {args.num_vars} variables..."
            )
        else:
            print(
                f"Generating {args.num_examples} formula string(s) with approx. {args.num_vars} variables..."
            )
        examples = make_nc1_examples(args.num_examples, args.num_vars, args.evaluate)

        if args.show_examples:
            print("\n--- Generated NC1 Examples ---")
            for i, ex in enumerate(examples):
                print(f"--- Example {i+1} ---")
                print(ex["text"])
    elif args.mode == "bfvp":
        config = {
            "num_examples": args.num_examples,
            "max_depth": args.max_depth,
            "max_ops": args.max_ops,
        }
        print("\nGenerating BFVP dataset with config:")
        pprint.pprint(config)
        examples = make_bfvp_examples(**config)

        if args.show_examples:
            print("\n--- Generated BFVP Examples ---")
            for ex in examples:
                print(ex["text"])
