#!/usr/bin/env python3
"""
Search for optimal BCD to 7-segment decoder circuit with 2, 3, and 4-input gates.
"""

import multiprocessing as mp
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import os
import time
import threading
import queue
import termios
import tty
import signal
import atexit

from bcd_optimization.solver import BCDTo7SegmentSolver
from bcd_optimization.truth_tables import SEGMENT_MINTERMS, SEGMENT_NAMES

# ANSI escape codes for terminal control
CLEAR_LINE = "\033[K"
MOVE_UP = "\033[A"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


class ProgressDisplay:
    """Async progress display that updates in a separate thread."""

    def __init__(self, stats_queue=None):
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.stats_queue = stats_queue
        self.original_termios = None

        # State
        self.completed = 0
        self.total = 0
        self.start_time = 0
        self.last_config = ""
        self.current_cost = 0
        self.message = ""

        # SAT solver stats (aggregated across all running configs)
        self.active_configs = {}  # config_str -> stats dict
        self.total_conflicts = 0
        self.total_decisions = 0
        self.total_vars = 0
        self.total_clauses = 0

    def _disable_input(self):
        """Disable keyboard echo and hide cursor."""
        try:
            self.original_termios = termios.tcgetattr(sys.stdin)
            new_termios = termios.tcgetattr(sys.stdin)
            new_termios[3] = new_termios[3] & ~termios.ECHO & ~termios.ICANON
            termios.tcsetattr(sys.stdin, termios.TCSANOW, new_termios)
        except (termios.error, AttributeError):
            pass
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()

    def _restore_input(self):
        """Restore keyboard echo and show cursor."""
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()
        if self.original_termios:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, self.original_termios)
            except (termios.error, AttributeError):
                pass
            self.original_termios = None

    def start(self, total, cost):
        """Start the progress display thread."""
        with self.lock:
            self.completed = 0
            self.total = total
            self.start_time = time.time()
            self.last_config = ""
            self.current_cost = cost
            self.message = ""
            self.active_configs = {}
            self.total_conflicts = 0
            self.total_decisions = 0
            self.running = True

        self._disable_input()
        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the progress display thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)
        self._restore_input()
        # Clear the line
        print(f"\r{CLEAR_LINE}", end="", flush=True)

    def update(self, completed, last_config=""):
        """Update progress state (called from main thread)."""
        with self.lock:
            self.completed = completed
            if last_config:
                self.last_config = last_config
                # Remove completed config from active
                if last_config in self.active_configs:
                    del self.active_configs[last_config]

    def set_message(self, msg):
        """Set a temporary message to display."""
        with self.lock:
            self.message = msg

    def _poll_stats(self):
        """Poll stats queue for updates from worker processes."""
        if not self.stats_queue:
            return

        try:
            while True:
                stats = self.stats_queue.get_nowait()
                config = stats.get('config', '')
                with self.lock:
                    self.active_configs[config] = stats
                    # Aggregate stats
                    self.total_conflicts = sum(s.get('conflicts', 0) for s in self.active_configs.values())
                    self.total_decisions = sum(s.get('decisions', 0) for s in self.active_configs.values())
                    self.total_vars = sum(s.get('vars', 0) for s in self.active_configs.values())
                    self.total_clauses = sum(s.get('clauses', 0) for s in self.active_configs.values())
        except:
            pass  # Queue empty

    def _update_loop(self):
        """Background thread that updates the display."""
        import shutil

        spinner = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        spin_idx = 0
        last_conflicts = 0
        conflict_rate = 0

        def fmt_num(n):
            if n >= 1_000_000:
                return f"{n/1_000_000:.1f}M"
            elif n >= 1_000:
                return f"{n/1_000:.1f}K"
            return str(int(n))

        while self.running:
            self._poll_stats()

            # Get terminal width each iteration (in case of resize)
            try:
                term_width = shutil.get_terminal_size().columns
            except:
                term_width = 80

            with self.lock:
                elapsed = time.time() - self.start_time
                speed = self.completed / elapsed if elapsed > 0 else 0
                remaining = self.total - self.completed
                eta = remaining / speed if speed > 0 else 0

                pct = 100 * self.completed / self.total if self.total > 0 else 0
                spin = spinner[spin_idx % len(spinner)]

                # Calculate conflict rate
                conflict_delta = self.total_conflicts - last_conflicts
                conflict_rate = conflict_rate * 0.7 + conflict_delta * 10 * 0.3
                last_conflicts = self.total_conflicts

                # Fixed parts (calculate their length)
                # Format: "  ⠹ [BAR] 2/6 (33%) 0.8/s [ACT] 29K"
                prefix = f"  {spin} ["
                count_str = f"] {self.completed}/{self.total} ({pct:.0f}%) {speed:.1f}/s"

                if self.total_conflicts > 0:
                    activity_level = min(max(conflict_rate / 50000, 0), 1.0)
                    conflict_str = f" {fmt_num(self.total_conflicts)}"
                else:
                    activity_level = 0
                    conflict_str = ""

                # Calculate available space for bars
                fixed_len = len(prefix) + len(count_str) + len(conflict_str) + 4  # +4 for " []" around activity
                available = term_width - fixed_len - 2  # -2 for safety margin

                if available < 10:
                    # Too narrow, minimal display
                    content = f"  {spin} {self.completed}/{self.total} {fmt_num(self.total_conflicts)}"
                else:
                    # Split available space: 70% progress bar, 30% activity bar
                    if self.total_conflicts > 0:
                        progress_width = int(available * 0.65)
                        activity_width = available - progress_width
                    else:
                        progress_width = available
                        activity_width = 0

                    # Progress bar
                    filled = int(progress_width * self.completed / self.total) if self.total > 0 else 0
                    bar = "█" * filled + "░" * (progress_width - filled)

                    # Activity bar
                    if activity_width > 0:
                        activity_filled = int(activity_width * activity_level)
                        activity_bar = f" [{CYAN}{'▮' * activity_filled}{'▯' * (activity_width - activity_filled)}{RESET}]{conflict_str}"
                    else:
                        activity_bar = ""

                    content = f"{prefix}{bar}{count_str}{activity_bar}"

                line = f"\r{content}{CLEAR_LINE}"
                sys.stdout.write(line)
                sys.stdout.flush()

            spin_idx += 1
            time.sleep(0.1)


def progress_bar(current, total, width=30, label=""):
    """Generate a progress bar string."""
    filled = int(width * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = 100 * current / total if total > 0 else 0
    return f"{label}[{bar}] {current}/{total} ({pct:.0f}%)"


def try_config_with_stats(n2, n3, n4, use_complements, restrict_functions, stats_queue):
    """Try a single (n2, n3, n4) configuration with stats reporting."""
    cost = n2 * 2 + n3 * 3 + n4 * 4
    n_gates = n2 + n3 + n4
    config_str = f"{n2}x2+{n3}x3+{n4}x4"

    if n_gates < 7:
        return None, cost, f"Skipped (only {n_gates} gates)", (n2, n3, n4)

    from pysat.formula import CNF
    from pysat.solvers import Solver

    solver = BCDTo7SegmentSolver()
    solver.generate_prime_implicants()

    start = time.time()
    try:
        # Build the CNF without solving
        cnf = solver._build_general_cnf(n2, n3, n4, use_complements, restrict_functions)
        if cnf is None:
            return None, cost, f"UNSAT (no valid config)", (n2, n3, n4)

        n_vars = cnf['n_vars']
        n_clauses = len(cnf['clauses'])

        # Report initial stats
        if stats_queue:
            try:
                stats_queue.put_nowait({
                    'config': config_str,
                    'phase': 'solving',
                    'vars': n_vars,
                    'clauses': n_clauses,
                    'conflicts': 0,
                    'decisions': 0,
                })
            except:
                pass

        # Solve with periodic stats updates
        with Solver(name='g3', bootstrap_with=CNF(from_clauses=cnf['clauses'])) as sat_solver:
            # Use solve_limited with conflict budget for progress updates
            conflict_budget = 10000
            total_conflicts = 0
            total_decisions = 0

            while True:
                sat_solver.conf_budget(conflict_budget)
                status = sat_solver.solve_limited()

                stats = sat_solver.accum_stats()
                total_conflicts = stats.get('conflicts', 0)
                total_decisions = stats.get('decisions', 0)

                if stats_queue:
                    try:
                        stats_queue.put_nowait({
                            'config': config_str,
                            'phase': 'solving',
                            'vars': n_vars,
                            'clauses': n_clauses,
                            'conflicts': total_conflicts,
                            'decisions': total_decisions,
                        })
                    except:
                        pass

                if status is not None:
                    # Solved (True = SAT, False = UNSAT)
                    break
                # status is None means budget exhausted, continue

            elapsed = time.time() - start

            if status:
                model = set(sat_solver.get_model())
                result = solver._decode_general_solution_from_cnf(model, cnf)
                return result, cost, f"SAT ({elapsed:.1f}s, {total_conflicts} conflicts)", (n2, n3, n4)
            else:
                return None, cost, f"UNSAT ({elapsed:.1f}s, {total_conflicts} conflicts)", (n2, n3, n4)

    except Exception as e:
        import traceback
        elapsed = time.time() - start
        return None, cost, f"Error ({elapsed:.1f}s): {e}", (n2, n3, n4)


def try_config(args):
    """Try a single (n2, n3, n4) configuration. Run in separate process."""
    n2, n3, n4, use_complements, restrict_functions, stats_queue = args
    return try_config_with_stats(n2, n3, n4, use_complements, restrict_functions, stats_queue)


def worker_init():
    """Initialize worker process to ignore SIGINT (parent handles it)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def verify_result(result):
    """Verify a synthesis result is correct."""
    def eval_func2(func, a, b):
        return (func >> (a * 2 + b)) & 1

    def eval_func3(func, a, b, c):
        return (func >> (a * 4 + b * 2 + c)) & 1

    def eval_func4(func, a, b, c, d):
        return (func >> (a * 8 + b * 4 + c * 2 + d)) & 1

    for digit in range(10):
        A = (digit >> 3) & 1
        B = (digit >> 2) & 1
        C = (digit >> 1) & 1
        D = digit & 1

        nodes = [A, B, C, D, 1-A, 1-B, 1-C, 1-D]

        for g in result.gates:
            if isinstance(g.input2, tuple):
                if len(g.input2) == 2:
                    k, l = g.input2
                    val = eval_func3(g.func, nodes[g.input1], nodes[k], nodes[l])
                else:
                    k, l, m = g.input2
                    val = eval_func4(g.func, nodes[g.input1], nodes[k], nodes[l], nodes[m])
            else:
                val = eval_func2(g.func, nodes[g.input1], nodes[g.input2])
            nodes.append(val)

        for seg in SEGMENT_NAMES:
            expected = 1 if digit in SEGMENT_MINTERMS[seg] else 0
            actual = nodes[result.output_map[seg]]
            if actual != expected:
                return False, f"Digit {digit}, {seg}: expected {expected}, got {actual}"

    return True, "All correct"


def cleanup_terminal():
    """Restore terminal to normal state."""
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.flush()


def main():
    # Register cleanup to ensure cursor is always restored
    atexit.register(cleanup_terminal)

    # Store original terminal settings for signal handler
    original_termios = None
    try:
        original_termios = termios.tcgetattr(sys.stdin)
    except (termios.error, AttributeError):
        pass

    # Create progress display early so signal handler can access it
    progress = ProgressDisplay(None)  # Queue set later
    executor_ref = [None]  # Mutable container for executor reference
    # State for signal handler status message
    search_state = {
        'start_time': 0,
        'group_start': 0,
        'configs_tested': 0,
        'current_cost': 0,
        'group_completed': 0,
        'group_size': 0,
    }

    def signal_handler(signum, frame):
        """Handle interrupt signals gracefully."""
        # Stop progress display first to prevent overwrites
        progress.running = False
        if progress.thread:
            progress.thread.join(timeout=0.2)
        # Shutdown executor without waiting
        if executor_ref[0]:
            executor_ref[0].shutdown(wait=False, cancel_futures=True)
        # Restore terminal
        sys.stdout.write(f"\r{CLEAR_LINE}")
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()
        if original_termios:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, original_termios)
            except (termios.error, AttributeError):
                pass
        # Print status message
        elapsed = time.time() - search_state['group_start'] if search_state['group_start'] else 0
        conflicts = progress.total_conflicts
        print(f"  {YELLOW}● Interrupted at {search_state['group_completed']}/{search_state['group_size']} configurations{RESET} ({elapsed:.1f}s, {conflicts} conflicts)")
        print(f"\n{YELLOW}Search interrupted by user{RESET}")
        os._exit(0)  # Hard exit to avoid cleanup errors from child processes

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}BCD to 7-Segment Optimal Circuit Search (with 4-input gates){RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print()
    print(f"Gates: AND, OR, XOR, XNOR, NAND, NOR (2, 3, and 4 input variants)")
    print(f"Primary input complements (A', B', C', D') are free")
    print()

    # Generate configurations sorted by cost
    configs = []
    for n2 in range(0, 12):
        for n3 in range(0, 8):
            for n4 in range(0, 6):
                cost = n2 * 2 + n3 * 3 + n4 * 4
                n_gates = n2 + n3 + n4
                # Need at least 7 gates for 7 outputs
                if 7 <= n_gates <= 11 and 14 <= cost <= 22:
                    configs.append((n2, n3, n4, True, True))

    # Sort by cost, then by number of gates
    configs.sort(key=lambda x: (x[0]*2 + x[1]*3 + x[2]*4, x[0]+x[1]+x[2]))

    # Group configs by cost
    cost_groups = {}
    for cfg in configs:
        cost = cfg[0] * 2 + cfg[1] * 3 + cfg[2] * 4
        if cost not in cost_groups:
            cost_groups[cost] = []
        cost_groups[cost].append(cfg)

    min_cost = min(cost_groups.keys())
    max_cost = max(cost_groups.keys())

    print(f"Searching {len(configs)} configurations from {min_cost} to {max_cost} inputs")
    print(f"Using {mp.cpu_count()} CPU cores")
    print()

    best_result = None
    best_cost = float('inf')

    total_start = time.time()
    search_state['start_time'] = total_start
    configs_tested = 0
    total_configs = len(configs)

    # Create shared queue for stats
    manager = Manager()
    stats_queue = manager.Queue()
    progress.stats_queue = stats_queue

    with ProcessPoolExecutor(max_workers=mp.cpu_count(), initializer=worker_init) as executor:
        executor_ref[0] = executor
        for cost in sorted(cost_groups.keys()):
            if cost >= best_cost:
                continue

            group = cost_groups[cost]
            group_size = len(group)
            group_start = time.time()
            completed_in_group = 0

            # Update state for signal handler
            search_state['current_cost'] = cost
            search_state['group_size'] = group_size
            search_state['group_completed'] = 0
            search_state['group_start'] = group_start

            print(f"\n{CYAN}{BOLD}Testing {cost} inputs{RESET} ({group_size} configurations)")
            print("-" * 50)

            # Start async progress display
            progress.start(group_size, cost)

            # Add stats_queue to each config
            configs_with_queue = [(cfg[0], cfg[1], cfg[2], cfg[3], cfg[4], stats_queue) for cfg in group]
            futures = {executor.submit(try_config, cfg): cfg for cfg in configs_with_queue}
            found_solution = False

            for future in as_completed(futures):
                cfg = futures[future]
                n2, n3, n4 = cfg[0], cfg[1], cfg[2]  # First 3 elements are gate counts
                completed_in_group += 1
                configs_tested += 1
                search_state['group_completed'] = completed_in_group
                search_state['configs_tested'] = configs_tested
                config_str = f"{n2}x2+{n3}x3+{n4}x4"

                try:
                    result, result_cost, status, _ = future.result(timeout=300)

                    # Update progress (always, even for UNSAT)
                    progress.update(completed_in_group, config_str)

                    if result is not None:
                        # Found a potential solution - stop progress to print
                        valid, msg = verify_result(result)
                        progress.stop()
                        if valid:
                            print(f"\n  {GREEN}✓ {n2}x2 + {n3}x3 + {n4}x4{RESET}: {status} {GREEN}(VERIFIED){RESET}")
                            if result_cost < best_cost:
                                best_result = result
                                best_cost = result_cost
                                found_solution = True
                                for f in futures:
                                    f.cancel()
                                break
                        else:
                            print(f"\n  {YELLOW}✗ {n2}x2 + {n3}x3 + {n4}x4{RESET}: INVALID - {msg}")
                            # Restart progress
                            progress.start(group_size, cost)
                            progress.update(completed_in_group, config_str)
                    # For UNSAT: just continue, progress bar updates automatically

                except Exception as e:
                    progress.stop()
                    print(f"\n  {n2}x2 + {n3}x3 + {n4}x4: Error - {e}")
                    progress.start(group_size, cost)
                    progress.update(completed_in_group)

            # Stop progress and print summary
            progress.stop()
            group_elapsed = time.time() - group_start

            if not found_solution:
                print(f"  {YELLOW}✗ All {group_size} configurations UNSAT{RESET} ({group_elapsed:.1f}s, {progress.total_conflicts} conflicts)")

            if best_result is not None and best_cost <= cost:
                print(f"\n{GREEN}{BOLD}Found solution at {best_cost} inputs!{RESET}")
                break

    total_elapsed = time.time() - total_start

    print()
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}RESULTS{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}")
    print(f"Search time: {total_elapsed:.1f} seconds ({configs_tested} configurations tested)")
    print(f"Average speed: {configs_tested/total_elapsed:.2f} configurations/second")

    if best_result:
        print(f"\n{GREEN}{BOLD}Best solution: {best_cost} gate inputs{RESET}")
        print()
        print("Gates:")

        node_names = ['A', 'B', 'C', 'D', "A'", "B'", "C'", "D'"]
        for g in best_result.gates:
            i1 = node_names[g.input1]
            if isinstance(g.input2, tuple):
                if len(g.input2) == 2:
                    k, l = g.input2
                    i2, i3 = node_names[k], node_names[l]
                    print(f"  g{g.index}: {g.func_name}({i1}, {i2}, {i3})")
                else:
                    k, l, m = g.input2
                    i2, i3, i4 = node_names[k], node_names[l], node_names[m]
                    print(f"  g{g.index}: {g.func_name}({i1}, {i2}, {i3}, {i4})")
            else:
                i2 = node_names[g.input2]
                print(f"  g{g.index}: {g.func_name}({i1}, {i2})")
            node_names.append(f"g{g.index}")

        print()
        print("Outputs:")
        for seg in SEGMENT_NAMES:
            print(f"  {seg} = {node_names[best_result.output_map[seg]]}")
    else:
        print(f"\n{YELLOW}No solution found in the search range (14-22 inputs).{RESET}")
        print("The minimum is likely 23 inputs (7x2 + 3x3 configuration).")


if __name__ == "__main__":
    main()
