"""
Verification module for BCD to 7-segment decoder synthesis results.

Ensures synthesized expressions produce correct outputs for all valid BCD inputs.
"""

from .truth_tables import SEGMENT_MINTERMS, SEGMENT_NAMES
from .quine_mccluskey import Implicant
from .solver import SynthesisResult


def evaluate_implicant(impl: Implicant, a: int, b: int, c: int, d: int) -> bool:
    """Evaluate an implicant on a specific input."""
    minterm = (a << 3) | (b << 2) | (c << 1) | d
    return impl.covers(minterm)


def evaluate_sop(implicants: list[Implicant], a: int, b: int, c: int, d: int) -> bool:
    """Evaluate a sum-of-products on a specific input (OR of AND terms)."""
    return any(evaluate_implicant(impl, a, b, c, d) for impl in implicants)


def verify_result(result: SynthesisResult) -> tuple[bool, list[str]]:
    """
    Verify that a synthesis result produces correct outputs for all BCD inputs.

    Args:
        result: The synthesis result to verify

    Returns:
        Tuple of (all_correct, list of error messages)
    """
    errors = []

    for segment in SEGMENT_NAMES:
        if segment not in result.implicants_by_output:
            continue

        implicants = result.implicants_by_output[segment]
        expected_on = set(SEGMENT_MINTERMS[segment])

        for digit in range(10):  # Valid BCD: 0-9
            a = (digit >> 3) & 1
            b = (digit >> 2) & 1
            c = (digit >> 1) & 1
            d = digit & 1

            actual = evaluate_sop(implicants, a, b, c, d)
            expected = digit in expected_on

            if actual != expected:
                errors.append(
                    f"Segment {segment}, digit {digit}: "
                    f"expected {expected}, got {actual}"
                )

    return len(errors) == 0, errors


def print_truth_table_comparison(result: SynthesisResult):
    """Print truth table comparing expected vs actual outputs."""
    print("Truth Table Verification")
    print("=" * 60)
    print(f"{'Digit':>5} | {'ABCD':>4} | Expected  | Actual    | Match")
    print("-" * 60)

    all_match = True

    for digit in range(10):
        a = (digit >> 3) & 1
        b = (digit >> 2) & 1
        c = (digit >> 1) & 1
        d = digit & 1

        expected = ""
        actual = ""
        match_str = ""

        for segment in SEGMENT_NAMES:
            exp = "1" if digit in SEGMENT_MINTERMS[segment] else "0"
            expected += exp

            if segment in result.implicants_by_output:
                implicants = result.implicants_by_output[segment]
                act = "1" if evaluate_sop(implicants, a, b, c, d) else "0"
            else:
                act = "?"

            actual += act
            match_str += "." if exp == act else "X"
            if exp != act:
                all_match = False

        print(f"{digit:>5} | {a}{b}{c}{d} | {expected:>9} | {actual:>9} | {match_str}")

    print("-" * 60)
    print(f"All correct: {all_match}")
    return all_match


if __name__ == "__main__":
    from .solver import BCDTo7SegmentSolver

    solver = BCDTo7SegmentSolver()
    result = solver.solve()

    print("\n")
    correct, errors = verify_result(result)

    if correct:
        print("Verification PASSED: All outputs correct!")
    else:
        print("Verification FAILED:")
        for err in errors:
            print(f"  {err}")

    print("\n")
    print_truth_table_comparison(result)
