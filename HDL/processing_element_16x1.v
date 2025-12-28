`timescale 1ns / 1ps

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Author: Ioannins Rizos
// Design Name: Floating Point 16x1 Processing Element
// Module Name: PE
//
// Aditional comments:
// This RTL code is part of a paper submission.
// Submitted for IEEE Transactions on Circuits and Systems for Artificial Intelligence.
// Code provided under the LISCENCE attached to this repository.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

module PE#(
    parameter integer NB  = 17,                                     // Bit width of operand B
    parameter integer NID =  7,                                     // Bit width of id section in B  [ FL(1 bit) | Unused (20-NID bits) | IDX (NID bits) | CMD (3 bits) | weight (1 bit) ] (make sure: NB bits total)
    parameter idx =  7'd0                                           // row of PE
    )(                      
    input                 clk,                                      // Clock input
    input   wire   [15:0] A_in,                                     // Input operand A (left,   activation)
    input   wire [NB-1:0] B_in,                                     // Input operand B (top,    weight/bias/control)
    output  wire   [15:0] C_out,                                    // Output product  (right,  activation)
    output  wire [NB-1:0] D_out                                     // Output product  (bottom, weight/bias/control)             
);

// Input registers
reg signed [15:0]      A_reg;                                       // operand A register
reg        [NB-1:0]    D_reg;                                       // operand B register

// State                        
localparam integer SC = 3;                                          // state count
localparam STATE_IDLE = 3'b001;                       
localparam STATE_LOAD = 3'b010;                       
localparam STATE_MULT = 3'b100;                       
reg  [SC-1:0]    State_reg;                                         // state register (one-hot)

// Control                      
localparam CTRL_RSET = 3'd1;                                        // synchronous reset from any state signal
localparam CTRL_ALT2 = 3'd2;                                        // two functions (select uint mode from idle / done from multiply)
localparam CTRL_INTM = 3'd3;                                        // select int mode from idle
localparam CTRL_LOAD = 3'd4;                                        // load from idle
localparam CTRL_MULT = 3'd5;                                        // multiply from idle
wire control_in = B_in[NB-1] & (B_in[3:1] != 0);                    // control signal received
wire [2:0] control_sig = B_in[3:1];                                 // the contol signal that was received

// loading indexes
wire weight_in = B_in[NB-1] & ~control_in;                          // weight input received
wire [NID:0]   IDX = B_in[NID+3:4];                                 // IDX part of B               

// multiplication logic                
reg            Weight_reg;                                          // weight register
        
wire [15:0]    A_xnored = {A_in[15] ^~ Weight_reg, A_in[14:0]};     // xnor is multiplication with weight
wire [15:0]    product;                                             // final product = xnor + bias           
fp_add #(.half(1)) add16 (.a_in(A_xnored), .b_in(B_in[15:0]), .result(product));

// outputs assignment                       
assign C_out = A_reg;                                               // pass A to the right
assign D_out = D_reg;                                               // pass weight/product/control down
            
// FSM                      
always @(posedge clk) begin                     
    if(control_in && (control_sig == CTRL_RSET)) begin              // if reset is received
        A_reg <= A_in;                                              // reset everything
        Weight_reg <= 0;
        D_reg <= B_in;
        State_reg <= STATE_IDLE;                                    // and go to IDLE
    end
    else begin                                                      // otherwise
        A_reg <= A_in;                                              // load inputs
        case(State_reg) 
            STATE_IDLE: begin                                       // at IDLE
                if(control_in &&
                       (control_sig == CTRL_LOAD)) begin            // else if load is received
                    State_reg <= STATE_LOAD;                        // go to LOAD state
                    D_reg <= B_in;                                  // and pass on the control signal                         
                end
                else if(control_in &&
                       (control_sig == CTRL_MULT)) begin            // else if mult is received
                    State_reg <= STATE_MULT;                        // go to MULT state
                    D_reg <= B_in;                                  // and pass on the control signal  
                end
                else D_reg <= 0;                                    // default: stay at idle and pass on 0 downwards
            end
            STATE_LOAD: begin                                       // at LOAD
                if(weight_in && IDX == idx) begin                   // if the correct indexe is found in top input
                    Weight_reg <= B_in[0];                          // set the weight from B_in
                    D_reg <= 0;                                     // pass on 0 downwards
                    State_reg <= STATE_IDLE;                        // and go to idle
                end
                else begin                                          // otherwise stay at LOAD
                    D_reg <= B_in;                                  // and pass the weight downwards
                end
            end
            STATE_MULT: begin                                       // at MULT
                if(control_in &&
                  (control_sig == CTRL_ALT2)) begin                 // if alt2 is received
                    State_reg <= STATE_IDLE;                        // go to idle
                    D_reg <= B_in;                                  // and pass on the control signal
                end
                else D_reg <= {1'b0, product};                      // otherwise pass on the current partial product
            end
            default: State_reg <= STATE_IDLE;                       // invalid state: go to IDLE
        endcase
    end
end

endmodule