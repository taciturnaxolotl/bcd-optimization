#!/usr/bin/env python3
"""
Search for optimal BCD to 7-segment decoder circuit.
Uses parallel SAT solving to search multiple configurations simultaneously.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import time

from bcd_optimization.solver import BCDTo7SegmentSolver
from bcd_optimization.truth_tables import SEGMENT_MINTERMS, SEGMENT_NAMES


def try_config(args):
    """Try a single (n2, n3) configuration. Run in separate process."""
    n2, n3, use_complements, restrict_functions = args
    cost = n2 * 2 + n3 * 3
    n_gates = n2 + n3

    if n_gates < 7:
        return None, cost, f"Skipped (only {n_gates} gates)"

    solver = BCDTo7SegmentSolver()
    solver.generate_prime_implicants()

    try:
        result = solver._try_mixed_synthesis(n2, n3, use_complements, restrict_functions)
        if result:
            return result, cost, "SUCCESS"
        else:
            return None, cost, "UNSAT"
    except Exception as e:
        return None, cost, f"Error: {e}"


def verify_result(result):
    """Verify a synthesis result is correct."""
    def eval_func2(func, a, b):
        return (func >> (a * 2 + b)) & 1

    def eval_func3(func, a, b, c):
        return (func >> (a * 4 + b * 2 + c)) & 1

    for digit in range(10):
        A = (digit >> 3) & 1
        B = (digit >> 2) & 1
        C = (digit >> 1) & 1
        D = digit & 1

        nodes = [A, B, C, D, 1-A, 1-B, 1-C, 1-D]

        for g in result.gates:
            if isinstance(g.input2, tuple):
                k, l = g.input2
                val = eval_func3(g.func, nodes[g.input1], nodes[k], nodes[l])
            else:
                val = eval_func2(g.func, nodes[g.input1], nodes[g.input2])
            nodes.append(val)

        for seg in SEGMENT_NAMES:
            expected = 1 if digit in SEGMENT_MINTERMS[seg] else 0
            actual = nodes[result.output_map[seg]]
            if actual != expected:
                return False, f"Digit {digit}, {seg}: expected {expected}, got {actual}"

    return True, "All correct"


def main():
    print("=" * 60)
    print("BCD to 7-Segment Optimal Circuit Search")
    print("=" * 60)
    print()
    print("Gates: AND, OR, XOR, NAND, NOR (2 and 3 input variants)")
    print("Primary input complements (A', B', C', D') are free")
    print()

    # Configurations to try, sorted by cost
    configs = []
    for n2 in range(0, 15):
        for n3 in range(0, 8):
            cost = n2 * 2 + n3 * 3
            n_gates = n2 + n3
            if 7 <= n_gates <= 12 and 18 <= cost <= 22:
                configs.append((n2, n3, True, True))

    # Sort by cost, then by number of gates
    configs.sort(key=lambda x: (x[0]*2 + x[1]*3, x[0]+x[1]))

    print(f"Searching {len(configs)} configurations from {min(c[0]*2+c[1]*3 for c in configs)} to {max(c[0]*2+c[1]*3 for c in configs)} inputs")
    print(f"Using {mp.cpu_count()} CPU cores")
    print()

    best_result = None
    best_cost = float('inf')

    start_time = time.time()

    # Group configs by cost for better progress reporting
    cost_groups = {}
    for cfg in configs:
        cost = cfg[0] * 2 + cfg[1] * 3
        if cost not in cost_groups:
            cost_groups[cost] = []
        cost_groups[cost].append(cfg)

    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        for cost in sorted(cost_groups.keys()):
            if cost >= best_cost:
                continue

            group = cost_groups[cost]
            print(f"Trying {cost} inputs ({len(group)} configurations)...", flush=True)

            futures = {executor.submit(try_config, cfg): cfg for cfg in group}

            for future in as_completed(futures):
                cfg = futures[future]
                n2, n3 = cfg[0], cfg[1]

                try:
                    result, result_cost, status = future.result(timeout=300)

                    if result is not None:
                        valid, msg = verify_result(result)
                        if valid:
                            print(f"  {n2}x2 + {n3}x3 = {result_cost}: {status} (verified)")
                            if result_cost < best_cost:
                                best_result = result
                                best_cost = result_cost
                                # Cancel remaining futures at this cost level
                                for f in futures:
                                    f.cancel()
                                break
                        else:
                            print(f"  {n2}x2 + {n3}x3 = {result_cost}: INVALID - {msg}")
                    else:
                        print(f"  {n2}x2 + {n3}x3 = {result_cost}: {status}")

                except Exception as e:
                    print(f"  {n2}x2 + {n3}x3: Error - {e}")

            if best_result is not None and best_cost <= cost:
                print(f"\nFound solution at {best_cost} inputs, stopping search.")
                break

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Search time: {elapsed:.1f} seconds")

    if best_result:
        print(f"Best solution: {best_cost} gate inputs")
        print()
        print("Gates:")

        node_names = ['A', 'B', 'C', 'D', "A'", "B'", "C'", "D'"]
        for g in best_result.gates:
            i1 = node_names[g.input1]
            if isinstance(g.input2, tuple):
                k, l = g.input2
                i2, i3 = node_names[k], node_names[l]
                print(f"  g{g.index}: {g.func_name}({i1}, {i2}, {i3})")
            else:
                i2 = node_names[g.input2]
                print(f"  g{g.index}: {g.func_name}({i1}, {i2})")
            node_names.append(f"g{g.index}")

        print()
        print("Outputs:")
        for seg in SEGMENT_NAMES:
            print(f"  {seg} = {node_names[best_result.output_map[seg]]}")
    else:
        print("No solution found in the search range.")


if __name__ == "__main__":
    main()
