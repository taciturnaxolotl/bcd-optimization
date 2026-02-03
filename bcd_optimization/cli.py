"""Command-line interface for BCD to 7-segment optimization."""

import argparse
import sys

from .solver import BCDTo7SegmentSolver
from .truth_tables import print_truth_table
from .export import to_verilog, to_c_code, to_equations


def main():
    parser = argparse.ArgumentParser(
        description="Optimize BCD to 7-segment decoder gate inputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bcd-optimize                    Run with default settings
  bcd-optimize --target 20        Try to beat 20 gate inputs
  bcd-optimize --exact            Use SAT-based exact synthesis
  bcd-optimize --truth-table      Show the BCD truth table
  bcd-optimize --format verilog   Output as Verilog module
  bcd-optimize --format c         Output as C function
        """,
    )

    parser.add_argument(
        "--target",
        type=int,
        default=23,
        help="Target gate input count to beat (default: 23)",
    )
    parser.add_argument(
        "--exact",
        action="store_true",
        help="Use SAT-based exact synthesis (slower but optimal)",
    )
    parser.add_argument(
        "--truth-table",
        action="store_true",
        help="Print the BCD to 7-segment truth table and exit",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["text", "verilog", "c", "equations"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.truth_table:
        print_truth_table()
        return 0

    # Suppress progress output for non-text formats
    quiet = args.format != "text"

    if not quiet:
        print("BCD to 7-Segment Decoder Optimizer")
        print("=" * 40)
        print(f"Target: < {args.target} gate inputs")
        print()

    solver = BCDTo7SegmentSolver()

    try:
        # Temporarily redirect stdout for quiet mode
        if quiet:
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

        result = solver.solve(target_cost=args.target, use_exact=args.exact)

        if quiet:
            sys.stdout = old_stdout

        # Output in requested format
        if args.format == "verilog":
            print(to_verilog(result))
        elif args.format == "c":
            print(to_c_code(result))
        elif args.format == "equations":
            print(to_equations(result))
        else:
            print()
            solver.print_result(result)

            if result.cost < args.target:
                print(f"\n✓ SUCCESS: Beat target by {args.target - result.cost} gate inputs!")
            else:
                print(f"\n✗ Did not beat target (need {result.cost - args.target + 1} more reduction)")

        return 0 if result.cost < args.target else 1

    except Exception as e:
        if quiet:
            sys.stdout = old_stdout
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
