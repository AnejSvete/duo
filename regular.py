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
    ) -> Tuple[Dict[str, int], List[List[int]], int, int]:
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

        num_elements = len(sorted_elements)  # This is the monoid size
        mult_table = [[0] * num_elements for _ in range(num_elements)]
        for i in range(num_elements):
            for j in range(num_elements):
                t_i, t_j = sorted_elements[i], sorted_elements[j]
                composed = tuple(t_i[state] for state in t_j)
                mult_table[i][j] = transform_to_id[composed]

        return symbol_to_monoid_id, mult_table, identity_id, num_elements


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


FSA_CREATORS = {
    "first_a": create_first_a_fsa,
    "first_aa": create_first_aa_fsa,
    "last_a": create_last_a_fsa,
    "last_aa": create_last_aa_fsa,
    "contains_a": create_contains_a_fsa,
    "contains_ab": create_contains_ab_fsa,
}


def make_fsa_examples(
    fsa_type, num_examples, min_log_len, max_log_len, mode
) -> Tuple[List[Dict[str, str]], int]:
    """Generates examples for a given FSA type and returns them with the monoid size."""
    fsa = FSA_CREATORS[fsa_type]()
    symbol_map, mult_table, identity_id, monoid_size = fsa.compute_syntactic_monoid()
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
        # START MODIFICATION: Restored empty_trace functionality
        elif mode == "empty_trace":
            if len(trace_levels) > 1:
                reduction_steps_list = trace_levels[1:]
                final_value = reduction_steps_list[-1]
                padded_steps = []
                # Replace intermediate steps with [PAD] tokens
                for step in reduction_steps_list[:-1]:
                    num_values = len(step.split())
                    padded_steps.append(" ".join(["[PAD]"] * num_values))

                if padded_steps:
                    # Join padded steps with the separator, then add the final value
                    padded_trace = " | ".join(padded_steps)
                    text = f"{initial_repr} # {padded_trace} | {final_value}"
                else:
                    # This case handles inputs so short they reduce in one step
                    text = f"{initial_repr} # {final_value}"
            else:
                # This case handles single-character inputs
                text = f"{initial_repr} # {trace_levels[0]}"
        # END MODIFICATION
        else:
            raise ValueError(f"Unknown format mode: {mode}")
        examples.append({"text": text})

    random.shuffle(examples)
    return examples, monoid_size


def get_monoid_size(fsa_type: str) -> int:
    """Pre-computes and returns the syntactic monoid size for an FSA type."""
    if fsa_type not in FSA_CREATORS:
        raise ValueError(f"Unknown FSA type: {fsa_type}")
    fsa = FSA_CREATORS[fsa_type]()
    _, _, _, monoid_size = fsa.compute_syntactic_monoid()
    return monoid_size
