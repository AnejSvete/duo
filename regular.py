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

    def accepts(self, input_string: str) -> bool:
        """Checks if the DFA accepts the given input string."""
        current_state = self.initial_state
        for symbol in input_string:
            if symbol not in self.alphabet:
                return False
            current_state = self.get_next_state(current_state, symbol)
            if current_state is None:
                return False
        return current_state in self.accepting_states

    def complement(self):
        """Returns the complement of this DFA."""
        comp = FiniteStateAutomaton(self.num_states, self.alphabet)
        comp.transitions = self.transitions.copy()
        comp.initial_state = self.initial_state
        comp.accepting_states = set(self.states) - self.accepting_states
        return comp

    def compute_syntactic_monoid(self) -> Tuple[Dict[str, int], List[List[int]], int]:
        """
        Computes the syntactic monoid, including the ID of the identity element.
        """
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
                composed = tuple(current_transform[i] for i in gen_transform)
                if composed not in monoid_elements:
                    monoid_elements.add(composed)
                    queue.append(composed)

        identity_transform = tuple(range(self.num_states))
        if identity_transform not in monoid_elements:
            monoid_elements.add(identity_transform)

        sorted_elements = sorted(list(monoid_elements))
        transform_to_id = {t: i for i, t in enumerate(sorted_elements)}
        identity_id = transform_to_id[identity_transform]

        symbol_to_monoid_id = {
            s: transform_to_id[t] for s, t in initial_transforms.items()
        }

        num_elements = len(sorted_elements)
        mult_table = [[0] * num_elements for _ in range(num_elements)]
        for i in range(num_elements):
            for j in range(num_elements):
                t_i, t_j = sorted_elements[i], sorted_elements[j]
                composed = tuple(t_i[state] for state in t_j)
                mult_table[i][j] = transform_to_id[composed]

        return symbol_to_monoid_id, mult_table, identity_id


def generate_random_dfa(num_states, alphabet_size):
    alphabet = [str(i) for i in range(alphabet_size)]
    dfa = FiniteStateAutomaton(num_states, alphabet)
    for state in dfa.states:
        for symbol in dfa.alphabet:
            dfa.set_transition(state, symbol, random.choice(dfa.states))
    for state in dfa.states:
        if random.random() < 0.5:
            dfa.set_accepting_state(state)
    return dfa


def create_parity_fsa():
    fsa = FiniteStateAutomaton(2, ["0", "1"])
    fsa.set_accepting_state(0)
    fsa.set_transition(0, "0", 0)
    fsa.set_transition(0, "1", 1)
    fsa.set_transition(1, "0", 1)
    fsa.set_transition(1, "1", 0)
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
    fsa = FiniteStateAutomaton(3, ["0", "1"])
    fsa.set_accepting_state(0)
    fsa.set_transition(0, "0", 0)
    fsa.set_transition(0, "1", 1)
    fsa.set_transition(1, "0", 2)
    fsa.set_transition(1, "1", 0)
    fsa.set_transition(2, "0", 1)
    fsa.set_transition(2, "1", 2)
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
    gen0 = (1, 2, 0, 3, 4)
    gen1 = (1, 2, 3, 4, 0)
    compose = lambda p1, p2: tuple(p1[p2[i]] for i in range(len(p2)))
    identity = tuple(range(5))
    elements = {identity}
    queue = [identity]
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        for g in [gen0, gen1]:
            neighbor = compose(curr, g)
            if neighbor not in elements:
                elements.add(neighbor)
                queue.append(neighbor)
    if len(elements) != 60:
        raise RuntimeError(f"Failed to generate A5. Generated {len(elements)}.")
    sorted_elements = sorted(list(elements))
    perm_to_id = {p: i for i, p in enumerate(sorted_elements)}
    fsa = FiniteStateAutomaton(60, ["0", "1"])
    for p, i in perm_to_id.items():
        fsa.set_transition(i, "0", perm_to_id[compose(p, gen0)])
        fsa.set_transition(i, "1", perm_to_id[compose(p, gen1)])
    fsa.initial_state = perm_to_id[identity]
    for i in range(60):
        fsa.set_accepting_state(i)
    return fsa


def get_monoid_trace(input_string, symbol_to_monoid_id, mult_table, identity_id):
    if not input_string:
        return []
    current_level = [symbol_to_monoid_id[s] for s in input_string]
    trace = []
    while len(current_level) > 1:
        trace.append(" ".join(map(str, current_level)))
        if len(current_level) % 2 != 0:
            current_level.append(identity_id)  # Pad with identity
        next_level = [
            mult_table[current_level[i]][current_level[i + 1]]
            for i in range(0, len(current_level), 2)
        ]
        current_level = next_level
    trace.append(" ".join(map(str, current_level)))
    return trace


def make_fsa_examples(fsa, num_examples, max_log_len, mode):
    symbol_map, mult_table, identity_id = fsa.compute_syntactic_monoid()
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

    # Generate a balanced set
    num_accepting = num_examples // 2
    num_rejecting = num_examples - num_accepting

    for i in range(num_examples):
        log_len = random.randint(1, max_log_len)
        length = 2**log_len
        target_fsa = fsa if i < num_accepting else fsa_complement
        # target_fsa = fsa
        input_string = sample_by_traversal(target_fsa, length)

        trace_levels = get_monoid_trace(
            input_string, symbol_map, mult_table, identity_id
        )
        initial_repr = " ".join(input_string)

        if mode == "trace":
            text = f"{initial_repr} # {' | '.join(trace_levels[1:])}"
        else:  # final_value
            text = f"{initial_repr} # {trace_levels[-1]}"
        examples.append({"text": text})

    random.shuffle(examples)
    return examples


def ascii_visualize_fsa(fsa):
    print("\n--- FSA Structure ---")
    for state in fsa.states:
        desc = f"State {state}"
        if state == fsa.initial_state:
            desc += " (Initial)"
        if state in fsa.accepting_states:
            desc += " (Accepting)"
        print(desc)
        for symbol in fsa.alphabet:
            print(f"  - '{symbol}' -> State {fsa.get_next_state(state, symbol)}")
    print("---------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate strings from an FSA and their syntactic monoid traces."
    )
    parser.add_argument(
        "--fsa_type",
        type=str,
        default="random",
        choices=["random", "a5", "parity", "ab_star", "mod_3", "same_start_end"],
        help="Type of FSA to generate.",
    )
    parser.add_argument(
        "--num_states", type=int, default=4, help="Number of states for random FSA."
    )
    parser.add_argument(
        "--alphabet_size", type=int, default=2, help="Alphabet size for random FSA."
    )
    parser.add_argument(
        "--max_log_len",
        type=int,
        default=4,
        help="Max log of string length (e.g., 4 -> max length 16).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="trace",
        choices=["trace", "final_value"],
        help="Output format.",
    )
    parser.add_argument(
        "--num_examples", type=int, default=10, help="Number of examples to generate."
    )
    parser.add_argument(
        "--show_examples", action="store_true", help="Print the generated examples."
    )
    parser.add_argument(
        "--show_fsa",
        action="store_true",
        help="Print an ASCII visualization of the FSA.",
    )
    args = parser.parse_args()

    fsa_map = {
        "a5": ("A5 group automaton", create_a5_fsa),
        "parity": ("Parity automaton", create_parity_fsa),
        "ab_star": ("(ab)* language automaton", create_ab_star_fsa),
        "mod_3": ("Modulo 3 automaton", create_mod_3_fsa),
        "same_start_end": (
            "Same Start/End Symbol automaton",
            create_same_start_end_fsa,
        ),
    }

    if args.fsa_type in fsa_map:
        name, func = fsa_map[args.fsa_type]
        print(f"Generating examples from the {name}.")
        fsa = func()
    else:
        print(f"Generating examples from a random FSA.")
        fsa = generate_random_dfa(args.num_states, args.alphabet_size)

    print(f"  - States: {fsa.num_states}, Alphabet: {fsa.alphabet}")
    print(
        f"  - Max String Length: 2^{args.max_log_len} = {2**args.max_log_len}, Format: {args.format}"
    )

    if args.show_fsa:
        ascii_visualize_fsa(fsa)

    examples = make_fsa_examples(fsa, args.num_examples, args.max_log_len, args.format)
    if args.show_examples:
        print("\n--- Generated Examples ---")
        for i, ex in enumerate(examples):
            print(f"[{i+1}] {ex['text']}")
