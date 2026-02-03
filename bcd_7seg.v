// BCD to 7-segment decoder (exact synthesis)
// 12 gates, 24 total gate inputs
// Method: exact_12gates

module bcd_to_7seg (
    input  wire [3:0] bcd,  // BCD input (0-9 valid)
    output wire [6:0] seg   // 7-segment output (a=seg[6], g=seg[0])
);

    // Input aliases
    wire A = bcd[3];
    wire B = bcd[2];
    wire C = bcd[1];
    wire D = bcd[0];

    // Internal gate outputs
    wire g0 = (A | ~D);
    wire g1 = (B ^ C);
    wire g2 = (C | ~g0);
    wire g3 = (g0 ^ g1);
    wire g4 = ~(A | g1);
    wire g5 = ~(B & g3);
    wire g6 = (C | g3);
    wire g7 = (~D & g6);
    wire g8 = ~(g4 & g5);
    wire g9 = (g3 | g7);
    wire g10 = ~(g2 & g5);
    wire g11 = (g3 | ~g9);

    // Segment output assignments
    assign seg[6] = g6;  // a
    assign seg[5] = g5;  // b
    assign seg[4] = g11;  // c
    assign seg[3] = g9;  // d
    assign seg[2] = g7;  // e
    assign seg[1] = g10;  // f
    assign seg[0] = g8;  // g

endmodule