"""
Pure Python implementation of Quine-McCluskey algorithm for Boolean minimization.

This implements multi-output prime implicant generation without requiring PyEDA.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=False)
class Implicant:
    """
    Represents a prime implicant with output coverage information.

    An implicant is represented by its mask and value:
    - mask: which bit positions matter (1 = matters, 0 = don't care)
    - value: the required bit values for positions that matter

    For 4 variables (A, B, C, D):
    - Bit 3 = A (MSB)
    - Bit 2 = B
    - Bit 1 = C
    - Bit 0 = D (LSB)
    """

    mask: int       # Which bits matter (1 = matters)
    value: int      # Required values for bits that matter
    output_mask: int = field(default=0, compare=False)
    covered_minterms: dict[str, set[int]] = field(default_factory=dict, compare=False)

    @property
    def num_literals(self) -> int:
        """Count the number of literals (gate inputs) in this implicant."""
        return bin(self.mask).count('1')

    def covers(self, minterm: int) -> bool:
        """Check if this implicant covers a given minterm."""
        return (minterm & self.mask) == (self.value & self.mask)

    def to_expr_str(self, var_names: list[str] = None) -> str:
        """Convert to a Boolean expression string (product term)."""
        if var_names is None:
            var_names = ['A', 'B', 'C', 'D']

        literals = []
        for i in range(4):
            bit = 1 << (3 - i)
            if self.mask & bit:
                if (self.value >> (3 - i)) & 1:
                    literals.append(var_names[i])
                else:
                    literals.append(f"{var_names[i]}'")

        return "".join(literals) if literals else "1"

    def __hash__(self):
        return hash((self.mask, self.value))

    def __eq__(self, other):
        if not isinstance(other, Implicant):
            return False
        return self.mask == other.mask and self.value == other.value

    def __repr__(self):
        return f"Implicant({self.to_expr_str()})"


def try_merge(impl1: Implicant, impl2: Implicant) -> Optional[Implicant]:
    """
    Try to merge two implicants differing in exactly one variable.

    Two implicants can merge if:
    1. They have the same mask
    2. They differ in exactly one bit position (within the mask)

    Returns new implicant with one less literal, or None if can't merge.
    """
    if impl1.mask != impl2.mask:
        return None

    diff = (impl1.value ^ impl2.value) & impl1.mask

    if bin(diff).count('1') != 1:
        return None

    new_mask = impl1.mask & ~diff
    new_value = impl1.value & new_mask

    return Implicant(mask=new_mask, value=new_value)


def quine_mccluskey(
    on_set: set[int],
    dc_set: set[int] = None,
    n_vars: int = 4
) -> list[Implicant]:
    """
    Run Quine-McCluskey algorithm to find all prime implicants.

    Args:
        on_set: Set of minterms where function is 1
        dc_set: Set of don't-care minterms (can be used for expansion)
        n_vars: Number of input variables

    Returns:
        List of prime implicants that cover at least one on-set minterm
    """
    if dc_set is None:
        dc_set = set()

    full_mask = (1 << n_vars) - 1

    # Start with on-set + don't-cares as initial implicants
    all_minterms = on_set | dc_set

    current = {}
    for m in all_minterms:
        impl = Implicant(mask=full_mask, value=m)
        current[(impl.mask, impl.value)] = impl

    prime_implicants = []

    while current:
        next_gen = {}
        used = set()

        impl_list = list(current.values())

        for i, impl1 in enumerate(impl_list):
            for j in range(i + 1, len(impl_list)):
                impl2 = impl_list[j]
                merged = try_merge(impl1, impl2)
                if merged:
                    key = (merged.mask, merged.value)
                    if key not in next_gen:
                        next_gen[key] = merged
                    used.add((impl1.mask, impl1.value))
                    used.add((impl2.mask, impl2.value))

        for key, impl in current.items():
            if key not in used:
                # Only keep if it covers at least one on-set minterm
                covers_on = any(impl.covers(m) for m in on_set)
                if covers_on:
                    prime_implicants.append(impl)

        current = next_gen

    return prime_implicants


def quine_mccluskey_multi_output(
    minterms_by_output: dict[str, set[int]],
    dc_set: set[int] = None,
    n_vars: int = 4
) -> list[Implicant]:
    """
    Generate prime implicants for multiple outputs with sharing tags.

    Generates primes for each output separately, then deduplicates and tags
    with output coverage. This correctly handles the case where different
    outputs have different on-sets.

    Args:
        minterms_by_output: Dict mapping output name to its on-set minterms
        dc_set: Set of don't-care minterms (shared across all outputs)
        n_vars: Number of input variables

    Returns:
        List of unique prime implicants with output coverage information
    """
    if dc_set is None:
        dc_set = set()

    # Generate prime implicants for each output separately
    all_primes = {}  # (mask, value) -> Implicant

    for output_name, on_set in minterms_by_output.items():
        primes = quine_mccluskey(on_set, dc_set, n_vars)
        for impl in primes:
            key = (impl.mask, impl.value)
            if key not in all_primes:
                all_primes[key] = Implicant(mask=impl.mask, value=impl.value)

    # Tag each prime with which outputs it can cover
    output_names = list(minterms_by_output.keys())
    result = []

    for impl in all_primes.values():
        impl.output_mask = 0
        impl.covered_minterms = {}

        for i, (name, minterms) in enumerate(minterms_by_output.items()):
            # An implicant can cover an output if:
            # 1. It covers some minterms in that output's on-set
            # 2. It doesn't cover any minterms in that output's off-set
            covered = {m for m in minterms if impl.covers(m)}

            # Check it doesn't cover any off-set minterms (0-9 that are not in on-set)
            off_set = set(range(10)) - minterms
            covers_off = any(impl.covers(m) for m in off_set)

            if covered and not covers_off:
                impl.covered_minterms[name] = covered
                impl.output_mask |= (1 << i)

        if impl.output_mask > 0:
            result.append(impl)

    return result


def greedy_cover(
    primes: list[Implicant],
    minterms_by_output: dict[str, set[int]]
) -> tuple[list[Implicant], int]:
    """
    Greedy set cover to select minimum-cost implicants.

    Returns:
        Tuple of (selected implicants, total cost)
    """
    uncovered = {
        (out, m)
        for out, minterms in minterms_by_output.items()
        for m in minterms
    }

    selected = []
    total_cost = 0

    while uncovered:
        best_impl = None
        best_ratio = -1
        best_covers = set()

        for impl in primes:
            if impl in selected:
                continue

            covers = set()
            for out, minterms in impl.covered_minterms.items():
                for m in minterms:
                    if (out, m) in uncovered:
                        covers.add((out, m))

            if not covers:
                continue

            cost = impl.num_literals if impl.num_literals > 0 else 1
            ratio = len(covers) / cost

            if ratio > best_ratio:
                best_ratio = ratio
                best_impl = impl
                best_covers = covers

        if best_impl is None:
            remaining = [(o, m) for o, m in uncovered]
            raise RuntimeError(f"Cannot cover: {remaining[:5]}...")

        selected.append(best_impl)
        total_cost += best_impl.num_literals
        uncovered -= best_covers

    return selected, total_cost


def print_prime_implicants(primes: list[Implicant]):
    """Debug helper to print all prime implicants."""
    print(f"Prime implicants ({len(primes)}):")
    for p in sorted(primes, key=lambda x: (-bin(x.output_mask).count('1'), x.num_literals)):
        outputs = list(p.covered_minterms.keys())
        print(f"  {p.to_expr_str():8} ({p.num_literals} lit) -> {', '.join(outputs)}")


if __name__ == "__main__":
    from .truth_tables import SEGMENT_MINTERMS, DONT_CARES, SEGMENT_NAMES

    minterms = {s: set(SEGMENT_MINTERMS[s]) for s in SEGMENT_NAMES}

    primes = quine_mccluskey_multi_output(
        minterms,
        set(DONT_CARES),
        n_vars=4
    )

    print_prime_implicants(primes)

    print("\nGreedy cover:")
    selected, cost = greedy_cover(primes, minterms)
    print(f"Selected {len(selected)} implicants, cost = {cost}")
    for impl in selected:
        print(f"  {impl.to_expr_str()}")
