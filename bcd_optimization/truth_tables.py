"""
Truth tables for BCD to 7-segment decoder.

BCD inputs: 4 bits (A, B, C, D) representing digits 0-9
- Positions 0-9: valid BCD digits
- Positions 10-15: don't care (invalid BCD)

7-segment display layout:
     aaa
    f   b
    f   b
     ggg
    e   c
    e   c
     ddd

Each segment's truth table: 1 = ON, 0 = OFF, - = don't care
"""

# Truth tables as strings (index 0-15, positions 10-15 are don't cares)
# Format: digit 0 at index 0, digit 9 at index 9, don't cares at 10-15
SEGMENT_TRUTH_TABLES = {
    'a': "1011011111------",  # ON for 0,2,3,5,6,7,8,9
    'b': "1111100111------",  # ON for 0,1,2,3,4,7,8,9
    'c': "1101111111------",  # ON for 0,1,3,4,5,6,7,8,9
    'd': "1011011011------",  # ON for 0,2,3,5,6,8,9
    'e': "1010001010------",  # ON for 0,2,6,8
    'f': "1000110111------",  # ON for 0,4,5,6,8,9
    'g': "0011111011------",  # ON for 2,3,4,5,6,8,9
}

SEGMENT_NAMES = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

# Minterms (ON-set) for each segment - these are the BCD digit indices where segment is ON
SEGMENT_MINTERMS = {
    'a': [0, 2, 3, 5, 6, 7, 8, 9],
    'b': [0, 1, 2, 3, 4, 7, 8, 9],
    'c': [0, 1, 3, 4, 5, 6, 7, 8, 9],
    'd': [0, 2, 3, 5, 6, 8, 9],
    'e': [0, 2, 6, 8],
    'f': [0, 4, 5, 6, 8, 9],
    'g': [2, 3, 4, 5, 6, 8, 9],
}

# Don't care positions (invalid BCD values 10-15)
DONT_CARES = [10, 11, 12, 13, 14, 15]

# Input variable names (MSB to LSB)
INPUT_VARS = ['A', 'B', 'C', 'D']


def minterm_to_bits(minterm: int) -> tuple[int, int, int, int]:
    """Convert a minterm index to its 4-bit representation (A, B, C, D)."""
    return (
        (minterm >> 3) & 1,  # A (MSB)
        (minterm >> 2) & 1,  # B
        (minterm >> 1) & 1,  # C
        minterm & 1,         # D (LSB)
    )


def bits_to_minterm(a: int, b: int, c: int, d: int) -> int:
    """Convert 4-bit representation to minterm index."""
    return (a << 3) | (b << 2) | (c << 1) | d


def print_truth_table():
    """Print the complete truth table for all segments."""
    print("BCD to 7-Segment Truth Table")
    print("=" * 50)
    print(f"{'Digit':>5} | {'A':>2} {'B':>2} {'C':>2} {'D':>2} | ", end="")
    print(" ".join(f"{s}" for s in SEGMENT_NAMES))
    print("-" * 50)

    for i in range(16):
        a, b, c, d = minterm_to_bits(i)
        digit = str(i) if i < 10 else "X"
        segments = " ".join(
            SEGMENT_TRUTH_TABLES[s][i] for s in SEGMENT_NAMES
        )
        print(f"{digit:>5} |  {a}  {b}  {c}  {d} | {segments}")


if __name__ == "__main__":
    print_truth_table()
