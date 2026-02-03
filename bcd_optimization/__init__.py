"""BCD to 7-segment decoder optimization using SAT-based exact synthesis."""

from .solver import BCDTo7SegmentSolver, SynthesisResult, CostBreakdown
from .truth_tables import SEGMENT_TRUTH_TABLES, SEGMENT_NAMES, SEGMENT_MINTERMS
from .quine_mccluskey import Implicant, quine_mccluskey, quine_mccluskey_multi_output
from .export import to_verilog, to_c_code, to_equations, to_dot
from .verify import verify_result

__all__ = [
    "BCDTo7SegmentSolver",
    "SynthesisResult",
    "SEGMENT_TRUTH_TABLES",
    "SEGMENT_NAMES",
    "SEGMENT_MINTERMS",
    "Implicant",
    "quine_mccluskey",
    "quine_mccluskey_multi_output",
    "to_verilog",
    "to_c_code",
    "to_equations",
    "to_dot",
    "verify_result",
]
__version__ = "0.1.0"
