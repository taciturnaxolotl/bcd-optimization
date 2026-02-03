"""
BCD to 7-segment decoder solver using SAT-based exact synthesis.

This module implements a multi-output logic synthesis solver that minimizes
gate inputs through shared term extraction and SAT/MaxSAT optimization.
"""

from dataclasses import dataclass, field
from typing import Optional
from pysat.formula import WCNF, CNF
from pysat.examples.rc2 import RC2
from pysat.solvers import Solver

from .truth_tables import SEGMENT_NAMES, SEGMENT_MINTERMS, DONT_CARES
from .quine_mccluskey import (
    Implicant,
    quine_mccluskey_multi_output,
    greedy_cover,
)


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for a synthesis result."""

    and_inputs: int      # Inputs to AND gates (multi-literal product terms only)
    or_inputs: int       # Inputs to OR gates (one per term per output)
    num_and_gates: int   # Number of AND gates (multi-literal terms)
    num_or_gates: int    # Number of OR gates (one per output = 7)

    @property
    def total(self) -> int:
        """Total gate inputs (AND + OR)."""
        return self.and_inputs + self.or_inputs


@dataclass
class SynthesisResult:
    """Result of logic synthesis optimization."""

    cost: int  # Total gate inputs (for backward compat, = cost_breakdown.and_inputs)
    implicants_by_output: dict[str, list[Implicant]]
    shared_implicants: list[tuple[Implicant, list[str]]]
    method: str
    expressions: dict[str, str] = field(default_factory=dict)
    cost_breakdown: CostBreakdown = None


class BCDTo7SegmentSolver:
    """
    Multi-output logic synthesis solver for BCD to 7-segment decoders.

    Uses a combination of:
    1. Quine-McCluskey with greedy cover for baseline
    2. MaxSAT optimization for minimum-cost covering with sharing
    3. SAT-based exact synthesis for provably optimal circuits
    """

    def __init__(self):
        self.prime_implicants: list[Implicant] = []
        self.minterms = {s: set(SEGMENT_MINTERMS[s]) for s in SEGMENT_NAMES}
        self.dc_set = set(DONT_CARES)

    def _compute_cost_breakdown(
        self,
        selected: list[Implicant],
        implicants_by_output: dict[str, list[Implicant]]
    ) -> CostBreakdown:
        """
        Compute detailed cost breakdown for a set of selected implicants.

        Cost model (assuming input complements are free):
        - AND gate inputs: Only for multi-literal terms (2+ literals)
          Single literals (A, B', etc.) are direct wires, no AND needed
        - OR gate inputs: One per term per output it feeds
        - AND gates: One per multi-literal term (shared across outputs)
        - OR gates: One per output (7 total)
        """
        and_inputs = 0
        num_and_gates = 0

        for impl in selected:
            if impl.num_literals >= 2:
                # Multi-literal term needs an AND gate
                and_inputs += impl.num_literals
                num_and_gates += 1
            # Single-literal terms are just wires (no AND gate cost)

        # OR inputs: count terms feeding each output
        or_inputs = sum(
            len(implicants_by_output[seg])
            for seg in SEGMENT_NAMES
            if seg in implicants_by_output
        )

        return CostBreakdown(
            and_inputs=and_inputs,
            or_inputs=or_inputs,
            num_and_gates=num_and_gates,
            num_or_gates=7,
        )

    def greedy_baseline(self) -> SynthesisResult:
        """
        Phase 1: Establish baseline using greedy set cover.

        Returns the baseline cost and selected implicants.
        """
        if not self.prime_implicants:
            self.generate_prime_implicants()

        selected, cost = greedy_cover(self.prime_implicants, self.minterms)

        # Organize by output
        implicants_by_output = {s: [] for s in SEGMENT_NAMES}
        shared = []

        for impl in selected:
            outputs_using = list(impl.covered_minterms.keys())
            if len(outputs_using) > 1:
                shared.append((impl, outputs_using))
            for out in outputs_using:
                implicants_by_output[out].append(impl)

        # Build expressions
        expressions = {}
        for segment in SEGMENT_NAMES:
            terms = [impl.to_expr_str() for impl in implicants_by_output[segment]]
            expressions[segment] = " + ".join(terms) if terms else "0"

        # Compute detailed cost breakdown
        cost_breakdown = self._compute_cost_breakdown(selected, implicants_by_output)

        return SynthesisResult(
            cost=cost_breakdown.total,  # Total = AND inputs + OR inputs
            implicants_by_output=implicants_by_output,
            shared_implicants=shared,
            method="greedy",
            expressions=expressions,
            cost_breakdown=cost_breakdown,
        )

    def generate_prime_implicants(self) -> list[Implicant]:
        """Generate all prime implicants with multi-output coverage tags."""
        self.prime_implicants = quine_mccluskey_multi_output(
            self.minterms,
            self.dc_set,
            n_vars=4
        )
        return self.prime_implicants

    def maxsat_optimize(self, target_cost: int = 22) -> SynthesisResult:
        """
        Phase 2: MaxSAT optimization for minimum-cost covering with sharing.

        Formulates the covering problem as weighted MaxSAT where:
        - Hard clauses: every minterm of every output must be covered
        - Soft clauses: minimize total literals (penalize each implicant)
        """
        if not self.prime_implicants:
            self.generate_prime_implicants()

        wcnf = WCNF()

        # Variable mapping: implicant index -> SAT variable (1-indexed)
        impl_vars = {i: i + 1 for i in range(len(self.prime_implicants))}

        # Hard constraints: every (output, minterm) pair must be covered
        for segment in SEGMENT_NAMES:
            for minterm in SEGMENT_MINTERMS[segment]:
                covering = []
                for i, impl in enumerate(self.prime_implicants):
                    if segment in impl.covered_minterms:
                        if minterm in impl.covered_minterms[segment]:
                            covering.append(impl_vars[i])

                if covering:
                    wcnf.append(covering)  # Hard: at least one must be selected
                else:
                    raise RuntimeError(
                        f"No implicant covers {segment}:{minterm}"
                    )

        # Soft constraints: penalize each implicant by its total gate input cost
        # Cost = AND inputs + OR inputs
        # - AND inputs: num_literals if >= 2, else 0 (single literals are wires)
        # - OR inputs: one per output this implicant covers
        for i, impl in enumerate(self.prime_implicants):
            and_cost = impl.num_literals if impl.num_literals >= 2 else 0
            or_cost = len(impl.covered_minterms)  # Number of outputs it feeds
            total_cost = and_cost + or_cost
            if total_cost > 0:
                wcnf.append([-impl_vars[i]], weight=total_cost)

        # Solve
        with RC2(wcnf) as solver:
            model = solver.compute()
            if model is None:
                raise RuntimeError("MaxSAT solver found no solution")

            # Extract selected implicants
            selected = []
            for i, impl in enumerate(self.prime_implicants):
                if impl_vars[i] in model:
                    selected.append(impl)

        # Organize by output
        implicants_by_output = {s: [] for s in SEGMENT_NAMES}
        shared = []

        for impl in selected:
            outputs_using = list(impl.covered_minterms.keys())
            if len(outputs_using) > 1:
                shared.append((impl, outputs_using))
            for out in outputs_using:
                implicants_by_output[out].append(impl)

        # Build expressions
        expressions = {}
        for segment in SEGMENT_NAMES:
            terms = [impl.to_expr_str() for impl in implicants_by_output[segment]]
            expressions[segment] = " + ".join(terms) if terms else "0"

        # Compute detailed cost breakdown
        cost_breakdown = self._compute_cost_breakdown(selected, implicants_by_output)

        return SynthesisResult(
            cost=cost_breakdown.total,  # Total = AND inputs + OR inputs
            implicants_by_output=implicants_by_output,
            shared_implicants=shared,
            method="maxsat",
            expressions=expressions,
            cost_breakdown=cost_breakdown,
        )

    def exact_synthesis(self, max_gates: int = 15) -> SynthesisResult:
        """
        Phase 3: SAT-based exact synthesis for provably optimal circuits.

        Encodes the circuit synthesis problem as SAT and iteratively searches
        for the minimum number of gates.
        """
        for num_gates in range(1, max_gates + 1):
            print(f"    Trying {num_gates} gates...")
            result = self._try_exact_synthesis(num_gates)
            if result is not None:
                return result

        raise RuntimeError(f"No solution found with up to {max_gates} gates")

    def _try_exact_synthesis(self, num_gates: int) -> Optional[SynthesisResult]:
        """
        Try to find a circuit with exactly num_gates gates.

        Uses a SAT encoding where:
        - Variables encode gate structure (which inputs each gate uses)
        - Variables encode gate function (AND, OR, NAND, NOR, etc.)
        - Constraints ensure functional correctness on all valid inputs
        """
        n_inputs = 4  # A, B, C, D
        n_outputs = 7  # a, b, c, d, e, f, g
        n_nodes = n_inputs + num_gates

        # Only verify on valid BCD inputs (0-9)
        truth_rows = list(range(10))
        n_rows = len(truth_rows)

        cnf = CNF()
        var_counter = [1]

        def new_var():
            v = var_counter[0]
            var_counter[0] += 1
            return v

        # Variables:
        # x[i][t] = output of node i on row t
        # s[i][j][k] = gate i uses inputs j and k
        # f[i][p][q] = gate i output when inputs are (p, q)
        # g[h][i] = output h comes from node i

        x = {}
        s = {}
        f = {}
        g = {}

        for i in range(n_nodes):
            x[i] = {t: new_var() for t in range(n_rows)}

        for i in range(n_inputs, n_nodes):
            s[i] = {}
            for j in range(i):
                s[i][j] = {k: new_var() for k in range(j + 1, i)}
            f[i] = {p: {q: new_var() for q in range(2)} for p in range(2)}

        for h in range(n_outputs):
            g[h] = {i: new_var() for i in range(n_nodes)}

        # Constraint 1: Primary inputs are fixed by truth table
        for t_idx, t in enumerate(truth_rows):
            for i in range(n_inputs):
                bit = (t >> (n_inputs - 1 - i)) & 1
                cnf.append([x[i][t_idx] if bit else -x[i][t_idx]])

        # Constraint 2: Each gate has exactly one input pair
        for i in range(n_inputs, n_nodes):
            all_sels = [s[i][j][k] for j in range(i) for k in range(j + 1, i)]
            # At least one
            cnf.append(all_sels)
            # At most one
            for idx1, sel1 in enumerate(all_sels):
                for sel2 in all_sels[idx1 + 1:]:
                    cnf.append([-sel1, -sel2])

        # Constraint 3: Gate function consistency
        for i in range(n_inputs, n_nodes):
            for j in range(i):
                for k in range(j + 1, i):
                    for t_idx in range(n_rows):
                        for pv in range(2):
                            for qv in range(2):
                                for outv in range(2):
                                    # If s[i][j][k] ∧ x[j][t]=pv ∧ x[k][t]=qv ∧ f[i][pv][qv]=outv
                                    # then x[i][t]=outv
                                    clause = [-s[i][j][k]]
                                    clause.append(-x[j][t_idx] if pv else x[j][t_idx])
                                    clause.append(-x[k][t_idx] if qv else x[k][t_idx])
                                    clause.append(-f[i][pv][qv] if outv else f[i][pv][qv])
                                    clause.append(x[i][t_idx] if outv else -x[i][t_idx])
                                    cnf.append(clause)

        # Constraint 4: Each output assigned to exactly one node
        for h in range(n_outputs):
            cnf.append([g[h][i] for i in range(n_nodes)])
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    cnf.append([-g[h][i], -g[h][j]])

        # Constraint 5: Output correctness
        for h, segment in enumerate(SEGMENT_NAMES):
            for t_idx, t in enumerate(truth_rows):
                expected = 1 if t in SEGMENT_MINTERMS[segment] else 0
                for i in range(n_nodes):
                    if expected:
                        cnf.append([-g[h][i], x[i][t_idx]])
                    else:
                        cnf.append([-g[h][i], -x[i][t_idx]])

        # Solve
        with Solver(bootstrap_with=cnf) as solver:
            if solver.solve():
                model = set(solver.get_model())
                return self._decode_exact_solution(
                    model, num_gates, n_inputs, n_nodes, x, s, f, g
                )
            return None

    def _decode_exact_solution(
        self, model, num_gates, n_inputs, n_nodes, x, s, f, g
    ) -> SynthesisResult:
        """Decode SAT solution into readable circuit description."""

        def is_true(var):
            return var in model

        node_names = ['A', 'B', 'C', 'D'] + [f'g{i}' for i in range(num_gates)]
        gate_exprs = {}

        for i in range(n_inputs, n_nodes):
            for j in range(i):
                for k in range(j + 1, i):
                    if is_true(s[i][j][k]):
                        # Decode gate function
                        func = 0
                        for p in range(2):
                            for q in range(2):
                                if is_true(f[i][p][q]):
                                    func |= (1 << (p * 2 + q))

                        op = self._decode_gate_function(func)
                        gate_exprs[i] = f"({node_names[j]} {op} {node_names[k]})"
                        node_names[i] = gate_exprs[i]
                        break

        expressions = {}
        for h, segment in enumerate(SEGMENT_NAMES):
            for i in range(n_nodes):
                if is_true(g[h][i]):
                    expressions[segment] = node_names[i]
                    break

        # For exact synthesis, all gates are 2-input gates
        # This is a different circuit topology than SOP
        cost_breakdown = CostBreakdown(
            and_inputs=num_gates * 2,  # All gates treated as "AND-like"
            or_inputs=0,               # No separate OR level in multi-level
            num_and_gates=num_gates,
            num_or_gates=0,
        )

        return SynthesisResult(
            cost=num_gates * 2,  # 2 inputs per 2-input gate
            implicants_by_output={},
            shared_implicants=[],
            method=f"exact_{num_gates}gates",
            expressions=expressions,
            cost_breakdown=cost_breakdown,
        )

    def _decode_gate_function(self, func: int) -> str:
        """Decode 4-bit function to gate type name."""
        # func[pq] gives output for inputs (p, q)
        # Bit 0: f(0,0), Bit 1: f(0,1), Bit 2: f(1,0), Bit 3: f(1,1)
        names = {
            0b0000: "0",
            0b0001: "AND",
            0b0010: "A>B",      # A AND NOT B
            0b0011: "A",
            0b0100: "B>A",      # B AND NOT A
            0b0101: "B",
            0b0110: "XOR",
            0b0111: "OR",
            0b1000: "NOR",
            0b1001: "XNOR",
            0b1010: "!B",
            0b1011: "A+!B",     # A OR NOT B
            0b1100: "!A",
            0b1101: "!A+B",     # NOT A OR B
            0b1110: "NAND",
            0b1111: "1",
        }
        return names.get(func, f"F{func:04b}")

    def solve(self, target_cost: int = 22, use_exact: bool = False) -> SynthesisResult:
        """
        Run the complete optimization pipeline.

        Args:
            target_cost: Target gate input count to beat
            use_exact: If True, use SAT-based exact synthesis (slower)

        Returns:
            Best synthesis result found
        """
        results = []

        # Phase 1: Generate primes and greedy baseline
        print("Phase 1: Generating prime implicants...")
        self.generate_prime_implicants()
        print(f"  Found {len(self.prime_implicants)} prime implicants")

        print("\nPhase 1b: Greedy set cover baseline...")
        greedy_result = self.greedy_baseline()
        results.append(greedy_result)
        print(f"  Greedy cost: {greedy_result.cost} gate inputs")

        # Phase 2: MaxSAT optimization
        print("\nPhase 2: MaxSAT optimization with sharing...")
        maxsat_result = self.maxsat_optimize(target_cost)
        results.append(maxsat_result)
        print(f"  MaxSAT cost: {maxsat_result.cost} gate inputs")
        print(f"  Shared terms: {len(maxsat_result.shared_implicants)}")

        # Phase 3: Exact synthesis (optional)
        if use_exact:
            print("\nPhase 3: SAT-based exact synthesis...")
            try:
                exact_result = self.exact_synthesis(max_gates=12)
                results.append(exact_result)
                print(f"  Exact cost: {exact_result.cost} gate inputs")
            except RuntimeError as e:
                print(f"  Exact synthesis failed: {e}")

        # Return best result
        best = min(results, key=lambda r: r.cost)
        print(f"\nBest result: {best.cost} gate inputs ({best.method})")

        return best

    def print_result(self, result: SynthesisResult):
        """Pretty-print a synthesis result."""
        print(f"\n{'=' * 60}")
        print(f"Synthesis Result: {result.method}")
        print(f"{'=' * 60}")

        if result.cost_breakdown:
            cb = result.cost_breakdown
            print(f"Cost breakdown:")
            print(f"  AND gate inputs: {cb.and_inputs} ({cb.num_and_gates} gates)")
            print(f"  OR gate inputs:  {cb.or_inputs} (7 gates)")
            print(f"  Total:           {cb.total} gate inputs")
        else:
            print(f"Total gate inputs: {result.cost}")

        if result.shared_implicants:
            print(f"\nShared terms ({len(result.shared_implicants)}):")
            for impl, outputs in result.shared_implicants:
                lit_info = f"({impl.num_literals} lit)" if impl.num_literals >= 2 else "(wire)"
                print(f"  {impl.to_expr_str():12} {lit_info:8} -> {', '.join(outputs)}")

        print("\nExpressions:")
        for segment in SEGMENT_NAMES:
            if segment in result.expressions:
                print(f"  {segment} = {result.expressions[segment]}")
