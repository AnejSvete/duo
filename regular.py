import argparse
import random
from typing import Dict, List, Tuple


class FiniteStateAutomaton:
    """A simple implementation of a Deterministic Finite State Automaton (DFA)."""

    def __init__(self, num_states: int, alphabet: List[str]):
        if num_states <= 0:
            raise ValueError("Number of states must be positive.")
        self.num_states = num_states
        self.alphabet = sorted(list(set(alphabet)))
        self.states = list(range(num_states))
        self.transitions = {}
        self.initial_state = 0
        self.accepting_states = set()

    def set_transition(self, src_state: int, symbol: str, dst_state: int):
        if src_state not in self.states or dst_state not in self.states:
            raise ValueError("Invalid state ID.")
        if symbol not in self.alphabet:
            raise ValueError(f"Symbol '{symbol}' is not in the alphabet.")
        self.transitions[(src_state, symbol)] = dst_state

    def set_accepting_state(self, state: int, is_accepting: bool = True):
        if state not in self.states:
            raise ValueError("Invalid state ID.")
        if is_accepting:
            self.accepting_states.add(state)
        elif state in self.accepting_states:
            self.accepting_states.remove(state)

    def get_next_state(self, current_state: int, symbol: str) -> int:
        return self.transitions.get((current_state, symbol))

    def complement(self):
        comp = FiniteStateAutomaton(self.num_states, self.alphabet)
        comp.transitions = self.transitions.copy()
        comp.initial_state = self.initial_state
        comp.accepting_states = set(self.states) - self.accepting_states
        return comp

    def compute_syntactic_monoid(
        self,
    ) -> Tuple[Dict[str, int], List[List[int]], int, int, Dict[int, tuple]]:
        """Computes the syntactic monoid and returns its components and size."""
        initial_transforms = {}
        for symbol in self.alphabet:
            transform = tuple(self.get_next_state(s, symbol) for s in self.states)
            if None in transform:
                raise RuntimeError(
                    f"Incomplete DFA: missing transition for symbol '{symbol}'"
                )
            initial_transforms[symbol] = transform

        monoid_elements = set(initial_transforms.values())
        queue = list(initial_transforms.values())
        head = 0
        while head < len(queue):
            current_transform = queue[head]
            head += 1
            for gen_transform in initial_transforms.values():
                # To process a string "uv", we apply u's transform (current)
                # then v's transform (gen). This is gen_transform ∘ current_transform.
                composed = tuple(gen_transform[i] for i in current_transform)
                if composed not in monoid_elements:
                    monoid_elements.add(composed)
                    queue.append(composed)

        identity_transform = tuple(range(self.num_states))
        if identity_transform not in monoid_elements:
            monoid_elements.add(identity_transform)

        sorted_elements = sorted(list(monoid_elements))
        transform_to_id = {t: i for i, t in enumerate(sorted_elements)}
        id_to_transform = {i: t for t, i in transform_to_id.items()}
        identity_id = transform_to_id[identity_transform]
        symbol_to_monoid_id = {
            s: transform_to_id[t] for s, t in initial_transforms.items()
        }

        num_elements = len(sorted_elements)
        mult_table = [[0] * num_elements for _ in range(num_elements)]
        for i in range(num_elements):
            for j in range(num_elements):
                t_i, t_j = sorted_elements[i], sorted_elements[j]
                # Multiplication table for t_i followed by t_j is t_j ∘ t_i.
                composed = tuple(t_j[state] for state in t_i)
                mult_table[i][j] = transform_to_id[composed]

        return (
            symbol_to_monoid_id,
            mult_table,
            identity_id,
            num_elements,
            id_to_transform,
        )


def get_monoid_trace(input_string, symbol_to_monoid_id, mult_table, identity_id):
    if not input_string:
        return []
    current_level = [symbol_to_monoid_id[s] for s in input_string]
    trace = []
    while len(current_level) > 0:
        trace.append(" ".join(map(str, current_level)))
        if len(current_level) == 1:
            break
        if len(current_level) % 2 != 0:
            current_level.append(identity_id)
        next_level = [
            mult_table[current_level[i]][current_level[i + 1]]
            for i in range(0, len(current_level), 2)
        ]
        current_level = next_level
    return trace


def create_parity_fsa():
    fsa = FiniteStateAutomaton(2, ["a", "b"])
    fsa.set_accepting_state(0)
    fsa.set_transition(0, "a", 0)
    fsa.set_transition(0, "b", 1)
    fsa.set_transition(1, "a", 1)
    fsa.set_transition(1, "b", 0)
    return fsa


def create_ab_star_fsa():
    fsa = FiniteStateAutomaton(3, ["a", "b"])
    fsa.set_accepting_state(0)
    fsa.set_transition(0, "a", 1)
    fsa.set_transition(0, "b", 2)
    fsa.set_transition(1, "a", 2)
    fsa.set_transition(1, "b", 0)
    fsa.set_transition(2, "a", 2)
    fsa.set_transition(2, "b", 2)
    return fsa


def create_mod_3_fsa():
    fsa = FiniteStateAutomaton(3, ["a", "b"])
    fsa.set_accepting_state(0)
    fsa.set_transition(0, "a", 0)
    fsa.set_transition(0, "b", 1)
    fsa.set_transition(1, "a", 2)
    fsa.set_transition(1, "b", 0)
    fsa.set_transition(2, "a", 1)
    fsa.set_transition(2, "b", 2)
    return fsa


def create_same_start_end_fsa():
    fsa = FiniteStateAutomaton(5, ["a", "b"])
    fsa.set_accepting_state(1)
    fsa.set_accepting_state(3)
    fsa.set_transition(0, "a", 1)
    fsa.set_transition(0, "b", 3)
    fsa.set_transition(1, "a", 1)
    fsa.set_transition(1, "b", 2)
    fsa.set_transition(2, "a", 1)
    fsa.set_transition(2, "b", 2)
    fsa.set_transition(3, "a", 4)
    fsa.set_transition(3, "b", 3)
    fsa.set_transition(4, "a", 4)
    fsa.set_transition(4, "b", 3)
    return fsa


def create_a5_fsa():
    gen_a = (1, 2, 0, 3, 4)
    gen_b = (1, 2, 3, 4, 0)
    # Note: This lambda composes right-to-left (p1 after p2), which is standard
    # for permutation group actions written on the left.
    compose = lambda p1, p2: tuple(p1[p2[i]] for i in range(len(p2)))
    identity = tuple(range(5))
    elements = {identity}
    queue = [identity]
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for g in [gen_a, gen_b]:
            # This is curr * g (right multiplication) which is compose(curr, g)
            neighbor = compose(curr, g)
            if neighbor not in elements:
                elements.add(neighbor)
                queue.append(neighbor)
    if len(elements) != 60:
        raise RuntimeError(f"Failed to generate A5. Generated {len(elements)}.")
    sorted_elements = sorted(list(elements))
    perm_to_id = {p: i for i, p in enumerate(sorted_elements)}
    fsa = FiniteStateAutomaton(60, ["a", "b"])
    for p, i in perm_to_id.items():
        # Transition from p on 'a' leads to state for p*gen_a
        fsa.set_transition(i, "a", perm_to_id[compose(p, gen_a)])
        fsa.set_transition(i, "b", perm_to_id[compose(p, gen_b)])
    fsa.initial_state = perm_to_id[identity]
    for i in range(60):
        fsa.set_accepting_state(i)
    return fsa


def create_first_a_fsa():
    fsa = FiniteStateAutomaton(3, ["a", "b"])
    fsa.set_accepting_state(1)
    fsa.set_transition(0, "a", 1)
    fsa.set_transition(0, "b", 2)
    fsa.set_transition(1, "a", 1)
    fsa.set_transition(1, "b", 1)
    fsa.set_transition(2, "a", 2)
    fsa.set_transition(2, "b", 2)
    return fsa


def create_first_aa_fsa():
    fsa = FiniteStateAutomaton(4, ["a", "b"])
    fsa.set_accepting_state(2)
    fsa.set_transition(0, "a", 1)
    fsa.set_transition(0, "b", 3)
    fsa.set_transition(1, "a", 2)
    fsa.set_transition(1, "b", 3)
    fsa.set_transition(2, "a", 2)
    fsa.set_transition(2, "b", 2)
    fsa.set_transition(3, "a", 3)
    fsa.set_transition(3, "b", 3)
    return fsa


def create_last_a_fsa():
    fsa = FiniteStateAutomaton(2, ["a", "b"])
    fsa.set_accepting_state(1)
    fsa.set_transition(0, "a", 1)
    fsa.set_transition(0, "b", 0)
    fsa.set_transition(1, "a", 1)
    fsa.set_transition(1, "b", 0)
    return fsa


def create_last_aa_fsa():
    fsa = FiniteStateAutomaton(3, ["a", "b"])
    fsa.set_accepting_state(2)
    fsa.set_transition(0, "a", 1)
    fsa.set_transition(0, "b", 0)
    fsa.set_transition(1, "a", 2)
    fsa.set_transition(1, "b", 0)
    fsa.set_transition(2, "a", 2)
    fsa.set_transition(2, "b", 0)
    return fsa


def create_contains_a_fsa():
    fsa = FiniteStateAutomaton(2, ["a", "b"])
    fsa.set_accepting_state(1)
    fsa.set_transition(0, "a", 1)
    fsa.set_transition(0, "b", 0)
    fsa.set_transition(1, "a", 1)
    fsa.set_transition(1, "b", 1)
    return fsa


def create_contains_ab_fsa():
    fsa = FiniteStateAutomaton(3, ["a", "b"])
    fsa.set_accepting_state(2)
    fsa.set_transition(0, "a", 1)
    fsa.set_transition(0, "b", 0)
    fsa.set_transition(1, "a", 1)
    fsa.set_transition(1, "b", 2)
    fsa.set_transition(2, "a", 2)
    fsa.set_transition(2, "b", 2)
    return fsa


def create_z60_fsa():
    """Creates an FSA representing the cyclic group Z_60."""
    num_elements = 60
    fsa = FiniteStateAutomaton(num_elements, ["a", "b"])

    # Let 'a' represent +1 mod 60 and 'b' represent +2 mod 60.
    # +1 is a generator, so the group is fully connected.
    gen_a_op = 1
    gen_b_op = 2

    for i in range(num_elements):
        # State i transitions to (i + op) % 60
        fsa.set_transition(i, "a", (i + gen_a_op) % num_elements)
        fsa.set_transition(i, "b", (i + gen_b_op) % num_elements)

    # Initial state corresponds to the identity element 0
    fsa.initial_state = 0

    # For modeling the group structure, all states are considered "accepting"
    for i in range(num_elements):
        fsa.set_accepting_state(i)

    return fsa


def create_a4_x_z5_fsa():
    """Creates an FSA representing the direct product group A4 x Z5."""
    # --- Part 1: Generate A4 (the alternating group on 4 elements) ---
    # Generators for A4: s = (0 1 2), t = (0 1)(2 3)
    gen_s = (1, 2, 0, 3)  # s(0)=1, s(1)=2, s(2)=0, s(3)=3
    gen_t = (1, 0, 3, 2)  # t(0)=1, t(1)=0, t(2)=3, t(3)=2

    # Permutation composition (p1 after p2, i.e., p1 o p2)
    # This corresponds to right multiplication: new_state = old_state * generator
    compose = lambda p1, p2: tuple(p1[p2[i]] for i in range(len(p2)))

    identity_a4 = tuple(range(4))
    elements_a4 = {identity_a4}
    queue = [identity_a4]
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for g in [gen_s, gen_t]:
            neighbor = compose(curr, g)
            if neighbor not in elements_a4:
                elements_a4.add(neighbor)
                queue.append(neighbor)

    if len(elements_a4) != 12:
        raise RuntimeError(f"Failed to generate A4. Generated {len(elements_a4)}.")

    sorted_elements_a4 = sorted(list(elements_a4))
    perm_to_id_a4 = {p: i for i, p in enumerate(sorted_elements_a4)}
    id_to_perm_a4 = {i: p for p, i in perm_to_id_a4.items()}

    # --- Part 2: Define the FSA for the direct product A4 x Z5 ---
    num_states = 12 * 5
    fsa = FiniteStateAutomaton(num_states, ["a", "b"])

    # Generators for A4 x Z5 are formed by pairing generators from A4 and Z5.
    # Let input 'a' correspond to group operation (*gen_s, +1)
    # Let input 'b' correspond to group operation (*gen_t, +2)
    gen_a_z5_op = 1
    gen_b_z5_op = 2

    for i in range(num_states):
        # Deconstruct state i into its A4 and Z5 components
        id_a4 = i // 5
        id_z5 = i % 5
        perm_a4 = id_to_perm_a4[id_a4]

        # --- Transition for input 'a' ---
        next_perm_a4_a = compose(perm_a4, gen_s)
        next_id_z5_a = (id_z5 + gen_a_z5_op) % 5
        # Reconstruct the next state's single integer ID
        next_state_id_a = perm_to_id_a4[next_perm_a4_a] * 5 + next_id_z5_a
        fsa.set_transition(i, "a", next_state_id_a)

        # --- Transition for input 'b' ---
        next_perm_a4_b = compose(perm_a4, gen_t)
        next_id_z5_b = (id_z5 + gen_b_z5_op) % 5
        # Reconstruct the next state's single integer ID
        next_state_id_b = perm_to_id_a4[next_perm_a4_b] * 5 + next_id_z5_b
        fsa.set_transition(i, "b", next_state_id_b)

    # Initial state corresponds to the identity element (identity_a4, 0)
    initial_id_a4 = perm_to_id_a4[identity_a4]
    fsa.initial_state = initial_id_a4 * 5 + 0

    # For modeling the group structure, all states are "accepting"
    for i in range(num_states):
        fsa.set_accepting_state(i)

    return fsa


FSA_CREATORS = {
    "a4_x_z5": create_a4_x_z5_fsa,
    "a5": create_a5_fsa,
    "ab_star": create_ab_star_fsa,
    "contains_a": create_contains_a_fsa,
    "contains_ab": create_contains_ab_fsa,
    "first_a": create_first_a_fsa,
    "first_aa": create_first_aa_fsa,
    "last_a": create_last_a_fsa,
    "last_aa": create_last_aa_fsa,
    "mod_3": create_mod_3_fsa,
    "parity": create_parity_fsa,
    "same_start_end": create_same_start_end_fsa,
    "z60": create_z60_fsa,
}


def make_fsa_examples(
    fsa: FiniteStateAutomaton,
    monoid_details: dict,
    num_examples: int,
    min_log_len: int,
    max_log_len: int,
    mode: str,
) -> List[Dict[str, str]]:
    """Generates examples for a given FSA and its computed monoid."""
    symbol_map = monoid_details["symbol_map"]
    mult_table = monoid_details["mult_table"]
    identity_id = monoid_details["identity_id"]

    fsa_complement = fsa.complement()
    examples = []

    def sample_by_traversal(target_fsa, length):
        string = []
        state = target_fsa.initial_state
        for _ in range(length):
            symbol = random.choice(target_fsa.alphabet)
            string.append(symbol)
            state = target_fsa.get_next_state(state, symbol)
        return "".join(string)

    num_accepting = num_examples // 2
    for i in range(num_examples):
        log_len = random.randint(min_log_len, max_log_len)
        length = 2**log_len
        target_fsa = fsa if i < num_accepting else fsa_complement
        input_string = sample_by_traversal(target_fsa, length)
        trace_levels = get_monoid_trace(
            input_string, symbol_map, mult_table, identity_id
        )
        initial_repr = " ".join(input_string)

        if mode == "trace":
            text = f"{initial_repr} # {' | '.join(trace_levels[1:])}"
        elif mode == "final_value":
            text = f"{initial_repr} # {trace_levels[-1]}"
        elif mode == "empty_trace":
            if len(trace_levels) > 1:
                reduction_steps_list = trace_levels[1:]
                final_value = reduction_steps_list[-1]
                padded_steps = []
                for step in reduction_steps_list[:-1]:
                    num_values = len(step.split())
                    padded_steps.append(" ".join(["[PAD]"] * num_values))
                if padded_steps:
                    padded_trace = " | ".join(padded_steps)
                    text = f"{initial_repr} # {padded_trace} | {final_value}"
                else:
                    text = f"{initial_repr} # {final_value}"
            else:
                text = f"{initial_repr} # {trace_levels[0]}"
        else:
            raise ValueError(f"Unknown format mode: {mode}")
        examples.append({"text": text})

    random.shuffle(examples)
    return examples


def get_monoid_size(fsa_type: str) -> int:
    if fsa_type not in FSA_CREATORS:
        raise ValueError(f"Unknown FSA type: {fsa_type}")
    fsa = FSA_CREATORS[fsa_type]()
    _, _, _, monoid_size = fsa.compute_syntactic_monoid()
    return monoid_size


def get_alphabet(fsa_type: str) -> List[str]:
    """Returns the alphabet for a given FSA type."""
    if fsa_type not in FSA_CREATORS:
        raise ValueError(f"Unknown FSA type: {fsa_type}")
    fsa = FSA_CREATORS[fsa_type]()
    return fsa.alphabet


def main():
    """Main function for running the FSA script from the command line."""
    parser = argparse.ArgumentParser(
        description="Generate examples from Finite State Automata syntactic monoids.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--fsa-type",
        type=str,
        required=True,
        choices=sorted(FSA_CREATORS.keys()),
        help="The type of Finite State Automaton to use.",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Total number of examples to generate.",
    )

    parser.add_argument(
        "--min-log-len",
        type=int,
        default=3,
        help="Minimum log2 of the input string length (e.g., 3 means length 2^3=8).",
    )

    parser.add_argument(
        "--max-log-len",
        type=int,
        default=3,
        help="Maximum log2 of the input string length (e.g., 5 means length 2^5=32).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="trace",
        choices=["trace", "final_value", "empty_trace"],
        help="Output format mode for the monoid reduction.",
    )

    parser.add_argument(
        "--debug-monoid",
        action="store_true",
        help="Print detailed information about the syntactic monoid's elements.",
    )

    args = parser.parse_args()

    print(f"--- 🚀 Running FSA: {args.fsa_type} 🚀 ---")

    try:
        # Create FSA and compute monoid details ONCE for efficiency
        fsa = FSA_CREATORS[args.fsa_type]()
        (
            symbol_map,
            mult_table,
            identity_id,
            monoid_size,
            id_to_transform,
        ) = fsa.compute_syntactic_monoid()

        print(f"Syntactic Monoid Size: {monoid_size}")
        print(f"Alphabet: {fsa.alphabet}")

        # --- New Debug Section ---
        if args.debug_monoid:
            print("\n--- 🔍 Monoid Details ---")
            print(
                f"Identity Element ID: {identity_id} -> {id_to_transform[identity_id]}"
            )
            print("\nAlphabet Symbol -> Monoid ID:")
            for symbol, monoid_id in sorted(symbol_map.items()):
                transform = id_to_transform[monoid_id]
                print(f"  '{symbol}' -> ID {monoid_id} {transform}")

            print("\nMonoid Elements (ID -> State Transformation):")
            # A transformation (t0, t1, ..) means state 0->t0, 1->t1, etc.
            header = " ".join([f"s{i}" for i in fsa.states])
            print(f"  ID | maps state to -> ({header})")
            print(f"  ---|------------------------")
            for elem_id, transform in sorted(id_to_transform.items()):
                transform_str = ", ".join(map(str, transform))
                print(f"  {elem_id:<2} | ({transform_str})")
            print("--------------------------")
        # -------------------------

        monoid_details = {
            "symbol_map": symbol_map,
            "mult_table": mult_table,
            "identity_id": identity_id,
        }

        print(
            f"\nGenerating {args.num_examples} examples "
            f"(length 2^{args.min_log_len} to 2^{args.max_log_len}) "
            f"with mode '{args.mode}'..."
        )

        examples = make_fsa_examples(
            fsa,
            monoid_details,
            args.num_examples,
            args.min_log_len,
            args.max_log_len,
            args.mode,
        )

        print("\n--- Generated Examples ---")
        for i, example in enumerate(examples):
            print(f"[{i+1}] {example['text']}")
        print("--------------------------")

    except (ValueError, RuntimeError) as e:
        print(f"\n--- ❌ Error ❌ ---")
        print(f"An error occurred: {e}")

    print("\n--- ✅ Script complete ---")


# --- Script Entry Point ---
if __name__ == "__main__":
    main()
