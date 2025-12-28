/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Author: Ioannins Rizos
// Design Name: Simple Floating point adder
// Module Name: fp_add
//
// Aditional comments:
//   - Combinational, single-cycle floating point adder
//   - Parameter `half` = 1 for 16-bit half-precision
//                       = 0 for 32-bit single-precision
// This RTL code is part of a paper submission.
// Submitted for IEEE Transactions on Circuits and Systems for Artificial Intelligence.
// Code provided under the LISCENCE attached to this repository.
// ============================================================

module fp_add #(
    parameter half = 0        // 0 = FP32, 1 = FP16
)(
    input  wire [31:0] a_in,
    input  wire [31:0] b_in,
    output wire [31:0] result
);

    // ------------------------------------------------------------
    // Parameters
    // ------------------------------------------------------------
    localparam WIDTH  = half ? 16 : 32;
    localparam EXPW   = half ? 5  : 8;
    localparam FRACW  = half ? 10 : 23;
    localparam EXPOFF = (1 << (EXPW-1)) - 1;

    // ------------------------------------------------------------
    // Extract: sign, exponent, fraction
    // ------------------------------------------------------------
    wire sign_a = a_in[WIDTH-1];
    wire sign_b = b_in[WIDTH-1];
    wire [EXPW-1:0]  exp_a  = a_in[WIDTH-2 -: EXPW];
    wire [EXPW-1:0]  exp_b  = b_in[WIDTH-2 -: EXPW];
    wire [FRACW-1:0] frac_a = a_in[FRACW-1:0];
    wire [FRACW-1:0] frac_b = b_in[FRACW-1:0];

    // ------------------------------------------------------------
    // Expand mantissa (restore hidden bit)
    // ------------------------------------------------------------
    wire [FRACW:0] mant_a = (exp_a == 0) ? {1'b0, frac_a} : {1'b1, frac_a};
    wire [FRACW:0] mant_b = (exp_b == 0) ? {1'b0, frac_b} : {1'b1, frac_b};

    // ------------------------------------------------------------
    // Full magnitude compare (correct)
    // ------------------------------------------------------------
    wire a_bigger =
        (exp_a > exp_b) ? 1'b1 :
        (exp_b > exp_a) ? 1'b0 :
        (mant_a > mant_b);

    wire [EXPW-1:0]  exp_large  = a_bigger ? exp_a  : exp_b;
    wire [EXPW-1:0]  exp_small  = a_bigger ? exp_b  : exp_a;
    wire [FRACW:0]   mant_large = a_bigger ? mant_a : mant_b;
    wire [FRACW:0]   mant_small = a_bigger ? mant_b : mant_a;
    wire             sign_large = a_bigger ? sign_a : sign_b;
    wire             sign_small = a_bigger ? sign_b : sign_a;

    // ------------------------------------------------------------
    // Align mantissa
    // ------------------------------------------------------------
    wire [EXPW:0] exp_diff = exp_large - exp_small;

    wire [FRACW+4:0] mant_large_s = {mant_large, 4'b0};
    wire [FRACW+4:0] mant_small_s = ({mant_small,4'b0} >> exp_diff);

    // ------------------------------------------------------------
    // Add or subtract
    // ------------------------------------------------------------
    wire [FRACW+5:0] raw_sum =
        (sign_large == sign_small) ?
             (mant_large_s + mant_small_s) :
             (mant_large_s - mant_small_s);

    // Result sign (large operand dominates for subtraction)
    wire sign_res = (sign_large == sign_small) ? sign_large : sign_large;

    // ------------------------------------------------------------
    // NORMALIZATION
    // ------------------------------------------------------------
    reg [FRACW+5:0] mant_norm;
    reg [EXPW:0] exp_norm;

    integer lead, shift, i;

    always @(*) begin
        lead = -1;
        shift = 0;

        // Cancellation → zero
        if (raw_sum == 0) begin
            mant_norm = 0;
            exp_norm  = 0;
        end

        // Mantissa overflow (leading bit beyond implicit position)
        else if (raw_sum[FRACW+5] == 1'b1) begin
            // shift RIGHT by 1
            mant_norm = raw_sum >> 1;
            exp_norm  = exp_large + 1;
        end

        // Normal case: find first '1'
        else begin
            mant_norm = raw_sum;
            exp_norm  = exp_large;

            // Find leading 1 (from MSB to LSB)
            
            for (i = FRACW+4; i >= 0; i=i-1)
                if (mant_norm[i] && (lead < 0))
                    lead = i;

            // shift LEFT so that leading 1 moves to FRACW position
            if (lead >= 0) begin
                shift = (FRACW+4) - lead;
                mant_norm = mant_norm << shift;
                exp_norm  = exp_norm - shift;
            end
        end
    end

    // ------------------------------------------------------------
    // ROUND: nearest-even
    // ------------------------------------------------------------
    wire guard  = mant_norm[3];
    wire roundb = mant_norm[2];
    wire sticky = |mant_norm[1:0];

    wire round_up = guard && (roundb | sticky | mant_norm[4]);

    wire [FRACW:0] rounded = {1'b0, mant_norm[FRACW+3:4]} + round_up;

    wire mant_ovf = rounded[FRACW];
    wire [FRACW-1:0] frac_res = mant_ovf ? rounded[FRACW:1] : rounded[FRACW-1:0];
    wire [EXPW-1:0]  exp_final = mant_ovf ? (exp_norm + 1) : exp_norm;

    // ------------------------------------------------------------
    // Pack
    // ------------------------------------------------------------
    wire [WIDTH-1:0] raw_res = {sign_res, exp_final, frac_res};
    assign result = half ? {16'b0, raw_res[15:0]} : raw_res;

endmodule