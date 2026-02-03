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


def to_dot(result: SynthesisResult, title: str = "BCD to 7-Segment Decoder") -> str:
    """
    Export synthesis result as Graphviz DOT format.

    Render with: dot -Tpng circuit.dot -o circuit.png
    Or:          dot -Tsvg circuit.dot -o circuit.svg

    Note: Inputs and their complements (A, A', B, B', C, C', D, D') are
    shown as free inputs - no inverter gates are needed.

    Args:
        result: The synthesis result
        title: Title for the diagram

    Returns:
        DOT source code as string
    """
    lines = []
    lines.append("digraph BCD_7Seg {")
    lines.append(f'    label="{title}\\n{result.cost} gate inputs, {len(result.shared_implicants)} shared terms";')
    lines.append('    labelloc="t";')
    lines.append('    fontsize=16;')
    lines.append('    rankdir=LR;')
    lines.append('    splines=ortho;')
    lines.append('    nodesep=0.3;')
    lines.append('    ranksep=0.8;')
    lines.append("")

    # Determine which inputs (true and complement) are actually used
    used_inputs = set()  # Will contain 'A', 'nA', 'B', 'nB', etc.

    for impl, _ in result.shared_implicants:
        for i, var in enumerate(['A', 'B', 'C', 'D']):
            bit = 1 << (3 - i)
            if impl.mask & bit:
                if (impl.value >> (3 - i)) & 1:
                    used_inputs.add(var)
                else:
                    used_inputs.add(f"n{var}")

    for segment in SEGMENT_NAMES:
        if segment not in result.implicants_by_output:
            continue
        for impl in result.implicants_by_output[segment]:
            for i, var in enumerate(['A', 'B', 'C', 'D']):
                bit = 1 << (3 - i)
                if impl.mask & bit:
                    if (impl.value >> (3 - i)) & 1:
                        used_inputs.add(var)
                    else:
                        used_inputs.add(f"n{var}")

    # Input nodes (true and complement forms are free)
    lines.append("    // Inputs (active high and low available for free)")
    lines.append('    subgraph cluster_inputs {')
    lines.append('        label="Inputs";')
    lines.append('        style=dashed;')
    lines.append('        color=gray;')
    for var in ['A', 'B', 'C', 'D']:
        if var in used_inputs:
            lines.append(f'        {var} [shape=circle, style=filled, fillcolor=lightblue, label="{var}"];')
        if f"n{var}" in used_inputs:
            lines.append(f'        n{var} [shape=circle, style=filled, fillcolor=lightcyan, label="{var}\'"];')
    lines.append('    }')
    lines.append("")

    # AND gates for shared product terms (only multi-literal terms need AND gates)
    # Single-literal terms are just wires from input to OR
    multi_literal_shared = [(i, impl, outputs) for i, (impl, outputs) in enumerate(result.shared_implicants) if impl.num_literals >= 2]
    single_literal_shared = [(i, impl, outputs) for i, (impl, outputs) in enumerate(result.shared_implicants) if impl.num_literals < 2]

    if multi_literal_shared:
        lines.append("    // Shared AND gates (multi-literal product terms)")
        lines.append('    subgraph cluster_and {')
        lines.append('        label="Product Terms";')
        lines.append('        style=dashed;')
        lines.append('        color=gray;')

        for i, impl, outputs in multi_literal_shared:
            term_label = impl.to_expr_str()
            lines.append(f'        and_{i} [shape=polygon, sides=4, style=filled, fillcolor=lightgreen, label="AND\\n{term_label}"];')
        lines.append('    }')
        lines.append("")

        # Connect inputs to AND gates
        lines.append("    // Input to AND connections")
        for i, impl, _ in multi_literal_shared:
            for j, var in enumerate(['A', 'B', 'C', 'D']):
                bit = 1 << (3 - j)
                if impl.mask & bit:
                    if (impl.value >> (3 - j)) & 1:
                        lines.append(f'    {var} -> and_{i};')
                    else:
                        lines.append(f'    n{var} -> and_{i};')
        lines.append("")

    # OR gates for outputs
    lines.append("    // Output OR gates")
    lines.append('    subgraph cluster_or {')
    lines.append('        label="Output OR Gates";')
    lines.append('        style=dashed;')
    lines.append('        color=gray;')
    for segment in SEGMENT_NAMES:
        lines.append(f'        or_{segment} [shape=ellipse, style=filled, fillcolor=lightsalmon, label="OR\\n{segment}"];')
    lines.append('    }')
    lines.append("")

    # Connect AND gates to OR gates (multi-literal shared terms)
    lines.append("    // AND to OR connections")
    for i, impl, outputs in multi_literal_shared:
        for segment in outputs:
            lines.append(f'    and_{i} -> or_{segment};')
    lines.append("")

    # Connect single-literal shared terms directly from inputs to OR gates
    if single_literal_shared:
        lines.append("    // Single-literal terms (direct wires)")
        for i, impl, outputs in single_literal_shared:
            for j, var in enumerate(['A', 'B', 'C', 'D']):
                bit = 1 << (3 - j)
                if impl.mask & bit:
                    src = var if (impl.value >> (3 - j)) & 1 else f"n{var}"
                    for segment in outputs:
                        lines.append(f'    {src} -> or_{segment};')
        lines.append("")

    # Handle non-shared terms (direct connections or inline ANDs)
    lines.append("    // Non-shared terms")
    nonshared_idx = 0
    for segment in SEGMENT_NAMES:
        if segment not in result.implicants_by_output:
            continue
        for impl in result.implicants_by_output[segment]:
            is_shared = any(impl == si for si, _ in result.shared_implicants)
            if is_shared:
                continue

            # Single literal - direct connection from input
            if impl.num_literals == 1:
                for j, var in enumerate(['A', 'B', 'C', 'D']):
                    bit = 1 << (3 - j)
                    if impl.mask & bit:
                        if (impl.value >> (3 - j)) & 1:
                            lines.append(f'    {var} -> or_{segment};')
                        else:
                            lines.append(f'    n{var} -> or_{segment};')
            else:
                # Multi-literal non-shared AND
                term_label = impl.to_expr_str()
                and_name = f"and_ns_{nonshared_idx}"
                lines.append(f'    {and_name} [shape=polygon, sides=4, style=filled, fillcolor=palegreen, label="AND\\n{term_label}"];')
                for j, var in enumerate(['A', 'B', 'C', 'D']):
                    bit = 1 << (3 - j)
                    if impl.mask & bit:
                        if (impl.value >> (3 - j)) & 1:
                            lines.append(f'    {var} -> {and_name};')
                        else:
                            lines.append(f'    n{var} -> {and_name};')
                lines.append(f'    {and_name} -> or_{segment};')
                nonshared_idx += 1
    lines.append("")

    # Output nodes
    lines.append("    // Outputs")
    lines.append('    subgraph cluster_outputs {')
    lines.append('        label="Outputs";')
    lines.append('        style=dashed;')
    lines.append('        color=gray;')
    for segment in SEGMENT_NAMES:
        lines.append(f'        out_{segment} [shape=doublecircle, style=filled, fillcolor=lightpink, label="{segment}"];')
    lines.append('    }')
    lines.append("")

    # Connect OR gates to outputs
    lines.append("    // OR to output connections")
    for segment in SEGMENT_NAMES:
        lines.append(f'    or_{segment} -> out_{segment};')

    lines.append("}")

    return "\n".join(lines)


def to_verilog_exact(result: SynthesisResult, module_name: str = "bcd_to_7seg") -> str:
    """
    Export exact synthesis result to Verilog.

    Args:
        result: The synthesis result with gates list populated
        module_name: Name for the Verilog module

    Returns:
        Verilog source code as string
    """
    if not result.gates:
        raise ValueError("No gates in result - use to_verilog for SOP results")

    lines = []
    lines.append(f"// BCD to 7-segment decoder (exact synthesis)")
    lines.append(f"// {len(result.gates)} gates, {result.cost} total gate inputs")
    lines.append(f"// Method: {result.method}")
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

    n_inputs = 4
    node_names = ['A', 'B', 'C', 'D']

    # Generate gate wires
    lines.append("    // Internal gate outputs")
    for gate in result.gates:
        node_names.append(f"g{gate.index}")
        in1 = node_names[gate.input1]
        in2 = node_names[gate.input2]
        expr = _gate_to_verilog_expr(gate.func, in1, in2)
        lines.append(f"    wire g{gate.index} = {expr};")

    lines.append("")
    lines.append("    // Segment output assignments")

    for i, segment in enumerate(SEGMENT_NAMES):
        if segment in result.output_map:
            node_idx = result.output_map[segment]
            src = node_names[node_idx]
            seg_idx = 6 - i  # a=seg[6], b=seg[5], ..., g=seg[0]
            lines.append(f"    assign seg[{seg_idx}] = {src};  // {segment}")

    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)


def _gate_to_verilog_expr(func: int, in1: str, in2: str) -> str:
    """Convert gate function code to Verilog expression."""
    # func encodes 2-input truth table: bit i = f(p,q) where i = p*2 + q
    # Bit 0: f(0,0), Bit 1: f(0,1), Bit 2: f(1,0), Bit 3: f(1,1)
    expressions = {
        0b0000: "1'b0",                       # constant 0
        0b0001: f"~({in1} | {in2})",          # NOR
        0b0010: f"(~{in1} & {in2})",          # B AND NOT A
        0b0011: f"~{in1}",                    # NOT A
        0b0100: f"({in1} & ~{in2})",          # A AND NOT B
        0b0101: f"~{in2}",                    # NOT B
        0b0110: f"({in1} ^ {in2})",           # XOR
        0b0111: f"~({in1} & {in2})",          # NAND
        0b1000: f"({in1} & {in2})",           # AND
        0b1001: f"~({in1} ^ {in2})",          # XNOR
        0b1010: in2,                          # B (pass through)
        0b1011: f"(~{in1} | {in2})",          # NOT A OR B
        0b1100: in1,                          # A (pass through)
        0b1101: f"({in1} | ~{in2})",          # A OR NOT B
        0b1110: f"({in1} | {in2})",           # OR
        0b1111: "1'b1",                       # constant 1
    }
    return expressions.get(func, f"/* unknown func {func} */")


def to_dot_exact(result: SynthesisResult, title: str = "BCD to 7-Segment Decoder") -> str:
    """
    Export exact synthesis result as Graphviz DOT format.

    Args:
        result: The synthesis result with gates list populated
        title: Title for the diagram

    Returns:
        DOT source code as string
    """
    if not result.gates:
        raise ValueError("No gates in result - use to_dot for SOP results")

    lines = []
    lines.append("digraph BCD_7Seg {")
    lines.append(f'    label="{title}\\n{len(result.gates)} gates, {result.cost} gate inputs";')
    lines.append('    labelloc="t";')
    lines.append('    fontsize=16;')
    lines.append('    rankdir=LR;')
    lines.append('    splines=ortho;')
    lines.append('    nodesep=0.5;')
    lines.append('    ranksep=1.0;')
    lines.append("")

    n_inputs = 4
    node_names = ['A', 'B', 'C', 'D']

    # Input nodes
    lines.append('    // Inputs')
    lines.append('    subgraph cluster_inputs {')
    lines.append('        label="Inputs";')
    lines.append('        style=dashed;')
    lines.append('        color=gray;')
    for name in node_names:
        lines.append(f'        {name} [shape=circle, style=filled, fillcolor=lightblue, label="{name}"];')
    lines.append('    }')
    lines.append("")

    # Gate nodes
    lines.append('    // Gates')
    lines.append('    subgraph cluster_gates {')
    lines.append('        label="Logic Gates";')
    lines.append('        style=dashed;')
    lines.append('        color=gray;')

    gate_colors = {
        'AND': 'lightgreen',
        'OR': 'lightsalmon',
        'XOR': 'lightyellow',
        'XNOR': 'lightyellow',
        'NAND': 'palegreen',
        'NOR': 'peachpuff',
    }

    for gate in result.gates:
        node_names.append(f"g{gate.index}")
        color = gate_colors.get(gate.func_name, 'lightgray')
        lines.append(f'        g{gate.index} [shape=box, style=filled, fillcolor={color}, label="{gate.func_name}"];')
    lines.append('    }')
    lines.append("")

    # Gate connections
    lines.append('    // Gate input connections')
    node_names_lookup = ['A', 'B', 'C', 'D'] + [f"g{g.index}" for g in result.gates]
    for gate in result.gates:
        in1 = node_names_lookup[gate.input1]
        in2 = node_names_lookup[gate.input2]
        lines.append(f'    {in1} -> g{gate.index};')
        lines.append(f'    {in2} -> g{gate.index};')
    lines.append("")

    # Output nodes
    lines.append('    // Outputs')
    lines.append('    subgraph cluster_outputs {')
    lines.append('        label="Segment Outputs";')
    lines.append('        style=dashed;')
    lines.append('        color=gray;')
    for segment in SEGMENT_NAMES:
        lines.append(f'        out_{segment} [shape=doublecircle, style=filled, fillcolor=lightpink, label="{segment}"];')
    lines.append('    }')
    lines.append("")

    # Output connections
    lines.append('    // Output connections')
    for segment in SEGMENT_NAMES:
        if segment in result.output_map:
            node_idx = result.output_map[segment]
            src = node_names_lookup[node_idx]
            lines.append(f'    {src} -> out_{segment};')

    lines.append("}")

    return "\n".join(lines)


def to_c_exact(result: SynthesisResult, func_name: str = "bcd_to_7seg") -> str:
    """
    Export exact synthesis result as C code.

    Args:
        result: The synthesis result with gates list populated
        func_name: Name for the C function

    Returns:
        C source code as string
    """
    if not result.gates:
        raise ValueError("No gates in result - use to_c_code for SOP results")

    lines = []
    lines.append("/*")
    lines.append(" * BCD to 7-segment decoder (exact synthesis)")
    lines.append(f" * {len(result.gates)} gates, {result.cost} total gate inputs")
    lines.append(f" * Method: {result.method}")
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
    lines.append("")

    node_names = ['A', 'B', 'C', 'D']

    lines.append("    // Gate outputs")
    for gate in result.gates:
        node_names.append(f"g{gate.index}")
        in1 = node_names[gate.input1]
        in2 = node_names[gate.input2]
        expr = _gate_to_c_expr(gate.func, in1, in2)
        lines.append(f"    uint8_t g{gate.index} = {expr};")

    lines.append("")
    lines.append("    // Pack segment outputs (bit 6 = a, bit 0 = g)")

    pack_parts = []
    for i, segment in enumerate(SEGMENT_NAMES):
        if segment in result.output_map:
            node_idx = result.output_map[segment]
            src = node_names[node_idx]
            pack_parts.append(f"({src} << {6-i})")

    lines.append(f"    return {' | '.join(pack_parts)};")
    lines.append("}")

    return "\n".join(lines)


def _gate_to_c_expr(func: int, in1: str, in2: str) -> str:
    """Convert gate function code to C expression."""
    # func encodes 2-input truth table: bit i = f(p,q) where i = p*2 + q
    expressions = {
        0b0000: "0",                          # constant 0
        0b0001: f"!({in1} | {in2})",          # NOR
        0b0010: f"(!{in1} & {in2})",          # B AND NOT A
        0b0011: f"!{in1}",                    # NOT A
        0b0100: f"({in1} & !{in2})",          # A AND NOT B
        0b0101: f"!{in2}",                    # NOT B
        0b0110: f"({in1} ^ {in2})",           # XOR
        0b0111: f"!({in1} & {in2})",          # NAND
        0b1000: f"({in1} & {in2})",           # AND
        0b1001: f"!({in1} ^ {in2})",          # XNOR
        0b1010: in2,                          # B (pass through)
        0b1011: f"(!{in1} | {in2})",          # NOT A OR B
        0b1100: in1,                          # A (pass through)
        0b1101: f"({in1} | !{in2})",          # A OR NOT B
        0b1110: f"({in1} | {in2})",           # OR
        0b1111: "1",                          # constant 1
    }
    return expressions.get(func, f"/* unknown func {func} */")


if __name__ == "__main__":
    from .solver import BCDTo7SegmentSolver

    solver = BCDTo7SegmentSolver()
    result = solver.solve()

    print("=" * 60)
    print("DOT OUTPUT (save as .dot, render with Graphviz)")
    print("=" * 60)
    print(to_dot(result))
