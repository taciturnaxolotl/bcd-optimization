/*
 * BCD to 7-segment decoder (exact synthesis)
 * 12 gates, 24 total gate inputs
 * Method: exact_12gates
 */

#include <stdint.h>

uint8_t bcd_to_7seg(uint8_t bcd) {
    // Extract individual bits
    uint8_t A = (bcd >> 3) & 1;
    uint8_t B = (bcd >> 2) & 1;
    uint8_t C = (bcd >> 1) & 1;
    uint8_t D = bcd & 1;

    // Gate outputs
    uint8_t g0 = (A | !D);
    uint8_t g1 = (B ^ C);
    uint8_t g2 = (C | !g0);
    uint8_t g3 = (g0 ^ g1);
    uint8_t g4 = !(A | g1);
    uint8_t g5 = !(B & g3);
    uint8_t g6 = (C | g3);
    uint8_t g7 = (!D & g6);
    uint8_t g8 = !(g4 & g5);
    uint8_t g9 = (g3 | g7);
    uint8_t g10 = !(g2 & g5);
    uint8_t g11 = (g3 | !g9);

    // Pack segment outputs (bit 6 = a, bit 0 = g)
    return (g6 << 6) | (g5 << 5) | (g11 << 4) | (g9 << 3) | (g7 << 2) | (g10 << 1) | (g8 << 0);
}