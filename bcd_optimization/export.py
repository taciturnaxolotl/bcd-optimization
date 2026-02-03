"""
Export synthesized circuits to various formats (Verilog, VHDL, etc).
"""

from .solver import SynthesisResult
from .truth_tables import SEGMENT_NAMES
from .quine_mccluskey import Implicant


def to_verilog(result: SynthesisResult, module_name: str = "bcd_to_7seg") -> str:
    """
    Export synthesis result to Verilog.

    Args:
        result: The synthesis result
        module_name: Name for the Verilog module

    Returns:
        Verilog source code as string
    """
    lines = []
    lines.append(f"// BCD to 7-segment decoder")
    lines.append(f"// Synthesized with {result.cost} gate inputs using {result.method}")
    lines.append(f"// Shared terms: {len(result.shared_implicants)}")
    lines.append("")
    lines.append(f"module {module_name} (")
    lines.append("    input  wire [3:0] bcd,  // BCD input (0-9 valid)")
    lines.append("    output wire [6:0] seg   // 7-segment output (a=seg[6], g=seg[0])")
    lines.append(");")
    lines.append("")
    lines.append("    // Input aliases")
    lines.append("    wire A = bcd[3];")
    lines.append("    wire B = bcd[2];")
    lines.append("    wire C = bcd[1];")
    lines.append("    wire D = bcd[0];")
    lines.append("")

    # Generate wire declarations for shared terms
    if result.shared_implicants:
        lines.append("    // Shared product terms")
        for i, (impl, outputs) in enumerate(result.shared_implicants):
            term_name = f"term_{i}"
            expr = impl_to_verilog(impl)
            lines.append(f"    wire {term_name} = {expr};  // used by {', '.join(outputs)}")
        lines.append("")

    # Generate output assignments
    lines.append("    // Segment outputs")
    for i, segment in enumerate(SEGMENT_NAMES):
        if segment in result.implicants_by_output:
            terms = []
            for impl in result.implicants_by_output[segment]:
                # Check if this is a shared term
                shared_idx = None
                for j, (shared_impl, _) in enumerate(result.shared_implicants):
                    if impl == shared_impl:
                        shared_idx = j
                        break

                if shared_idx is not None:
                    terms.append(f"term_{shared_idx}")
                else:
                    terms.append(impl_to_verilog(impl))

            expr = " | ".join(terms) if terms else "1'b0"
            seg_idx = 6 - i  # a=seg[6], b=seg[5], ..., g=seg[0]
            lines.append(f"    assign seg[{seg_idx}] = {expr};  // {segment}")

    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)


def impl_to_verilog(impl: Implicant) -> str:
    """Convert an implicant to a Verilog expression."""
    var_names = ['A', 'B', 'C', 'D']
    terms = []

    for i in range(4):
        bit = 1 << (3 - i)
        if impl.mask & bit:
            if (impl.value >> (3 - i)) & 1:
                terms.append(var_names[i])
            else:
                terms.append(f"~{var_names[i]}")

    if not terms:
        return "1'b1"
    elif len(terms) == 1:
        return terms[0]
    else:
        return "(" + " & ".join(terms) + ")"


def to_equations(result: SynthesisResult) -> str:
    """
    Export synthesis result as Boolean equations.

    Args:
        result: The synthesis result

    Returns:
        Human-readable Boolean equations
    """
    lines = []
    lines.append(f"BCD to 7-Segment Decoder Equations")
    lines.append(f"Method: {result.method}")
    lines.append(f"Total gate inputs: {result.cost}")
    lines.append(f"Shared terms: {len(result.shared_implicants)}")
    lines.append("")

    if result.shared_implicants:
        lines.append("Shared product terms:")
        for impl, outputs in result.shared_implicants:
            lines.append(f"  {impl.to_expr_str():12} -> {', '.join(outputs)}")
        lines.append("")

    lines.append("Output equations:")
    for segment in SEGMENT_NAMES:
        if segment in result.expressions:
            lines.append(f"  {segment} = {result.expressions[segment]}")

    return "\n".join(lines)


def to_c_code(result: SynthesisResult, func_name: str = "bcd_to_7seg") -> str:
    """
    Export synthesis result as C code.

    Args:
        result: The synthesis result
        func_name: Name for the C function

    Returns:
        C source code as string
    """
    lines = []
    lines.append("/*")
    lines.append(" * BCD to 7-segment decoder")
    lines.append(f" * Synthesized with {result.cost} gate inputs using {result.method}")
    lines.append(" */")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append(f"uint8_t {func_name}(uint8_t bcd) {{")
    lines.append("    // Extract individual bits")
    lines.append("    uint8_t A = (bcd >> 3) & 1;")
    lines.append("    uint8_t B = (bcd >> 2) & 1;")
    lines.append("    uint8_t C = (bcd >> 1) & 1;")
    lines.append("    uint8_t D = bcd & 1;")
    lines.append("    uint8_t nA = !A, nB = !B, nC = !C, nD = !D;")
    lines.append("")

    # Generate shared terms
    if result.shared_implicants:
        lines.append("    // Shared product terms")
        for i, (impl, _) in enumerate(result.shared_implicants):
            expr = impl_to_c(impl)
            lines.append(f"    uint8_t t{i} = {expr};")
        lines.append("")

    # Generate output bits
    lines.append("    // Compute segment outputs")
    segment_exprs = []
    for seg_idx, segment in enumerate(SEGMENT_NAMES):
        if segment in result.implicants_by_output:
            terms = []
            for impl in result.implicants_by_output[segment]:
                shared_idx = None
                for j, (shared_impl, _) in enumerate(result.shared_implicants):
                    if impl == shared_impl:
                        shared_idx = j
                        break

                if shared_idx is not None:
                    terms.append(f"t{shared_idx}")
                else:
                    terms.append(impl_to_c(impl))

            expr = " | ".join(terms) if terms else "0"
            lines.append(f"    uint8_t {segment} = {expr};")
            segment_exprs.append(segment)

    lines.append("")
    lines.append("    // Pack into result (bit 6 = a, bit 0 = g)")
    pack_expr = " | ".join(
        f"({segment} << {6-i})"
        for i, segment in enumerate(SEGMENT_NAMES)
    )
    lines.append(f"    return {pack_expr};")
    lines.append("}")

    return "\n".join(lines)


def impl_to_c(impl: Implicant) -> str:
    """Convert an implicant to a C expression."""
    var_map = {
        'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D',
        "A'": 'nA', "B'": 'nB', "C'": 'nC', "D'": 'nD',
    }
    var_names = ['A', 'B', 'C', 'D']
    terms = []

    for i in range(4):
        bit = 1 << (3 - i)
        if impl.mask & bit:
            if (impl.value >> (3 - i)) & 1:
                terms.append(var_names[i])
            else:
                terms.append(f"n{var_names[i]}")

    if not terms:
        return "1"
    elif len(terms) == 1:
        return terms[0]
    else:
        return "(" + " & ".join(terms) + ")"


if __name__ == "__main__":
    from .solver import BCDTo7SegmentSolver

    solver = BCDTo7SegmentSolver()
    result = solver.solve()

    print("=" * 60)
    print("VERILOG OUTPUT")
    print("=" * 60)
    print(to_verilog(result))

    print("\n")
    print("=" * 60)
    print("C CODE OUTPUT")
    print("=" * 60)
    print(to_c_code(result))
