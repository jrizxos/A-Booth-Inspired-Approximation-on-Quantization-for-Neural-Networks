`timescale 1ns / 1ps

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Author: Ioannins Rizos
// Design Name: ASP2 Multiply Accumulate unit
// Module Name: ASP2
//
// Aditional comments: This RTL code is part of the paper submission: I. Rizos, G. Papatheodorou, A. Efthymiou "TODO Title"
// Submitted for TODO Journal. Code provided under the LISCENCE attached to this repository.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

module ASP2(     
    input   wire        clk,                                        // clock input       
    input   wire        rst,                                        // reset input          
    input   wire  [7:0] A_in,                                       // Input operand A (activation)
    input   wire  [5:0] B_in,                                       // Input operand B (weight in ASP2 form)
    input   wire        mode,                                       // (0: A uint, 1: A int)
    output  wire [31:0] C_out                                       // Output product    
);

// multiplication logic                 
wire A_ext_bit = A_in[7] & mode;                                    // extension bit: uint mode = 0 / int mode = sign of A
wire [15:0] A_extended = {{8{A_ext_bit}}, A_in};                    // extended A accoring to mode 

wire [2:0] sft_amm_p = B_in[5:3];                                   // positive shift ammount                 
wire [2:0] sft_amm_n = B_in[2:0];                                   // negative shift ammount                

wire [15:0] pp_p = A_extended << sft_amm_p;                         // positive partial product             
wire [15:0] pp_n = A_extended << sft_amm_n;                         // negative partial product                   

wire signed [15:0] subtraction = pp_p - pp_n;                       // subtraction of partial products 

// accumulation logic
reg signed [31:0] accumulator_reg;
always @(posedge clk) begin
    if(rst) begin
        accumulator_reg = 0;
    end
    else begin
        accumulator_reg = accumulator_reg + subtraction;
    end
end

assign C_out = accumulator_reg;                                     // output result

endmodule