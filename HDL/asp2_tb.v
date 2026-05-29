`timescale 1ns / 1ps

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Author: Ioannins Rizos
// Design Name: ASP2 Multiply Accumulate unit testbench
// Module Name: ASP2_tb
//
// Aditional comments: This RTL code is part of the paper submission: I. Rizos, G. Papatheodorou, A. Efthymiou "TODO Title"
// Submitted for TODO Journal. Code provided under the LISCENCE attached to this repository.
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

module ASP2_tb();

    // interface
    reg clk = 0;
    reg rst;
    reg [7:0] A_in;  
    reg [5:0] B_in;  
    reg mode;  
    wire [31:0] C_out;

    ASP2 DUT (.clk(clk), .rst(rst), .A_in(A_in), .B_in(B_in), .mode(mode), .C_out(C_out));

    always @(*) #1 clk <= !clk; 

    // test bench
    initial begin
        $dumpfile("test.vcd");              // for iverilog waveform output
        $dumpvars(0,ASP2_tb);
        
        #2; 
        mode <= 0;                          // unsigned A mode
        rst <= 1;
        A_in <= 0;                          // 0x0 = 0
        B_in <= map_asp2(0);
        
        #2;
        rst <= 0;
        A_in <= 255;                        // 255x1 = 255
        B_in <= map_asp2(1);

        #2;
        A_in <= 73;                         // 73x2 = 146
        B_in <= map_asp2(2);

        #2;
        A_in <= 42;                         // 42x-2 = -84
        B_in <= map_asp2(-2);

        #2;
        A_in <= 255;                        // 255x127 = 32,385
        B_in <= map_asp2(127);





        #2; 
        rst <= 1;
        A_in <= 0;                          // 0x0 = 0
        B_in <= map_asp2(0);
        mode <= 1;                          // signed A mode
        
        #2;
        rst <= 0;
        A_in <= -16;                        // -16x1 = -16
        B_in <= map_asp2(1);

        #2;
        A_in <= -40;                        // -40x80 = -3200
        B_in <= map_asp2(80);

        #2;
        A_in <= 127;                        // 127x127 = 16,129
        B_in <= map_asp2(127);

        #2;
        A_in <= -127;                       // -127x127 = -16,129
        B_in <= map_asp2(127);

        #5;
        $finish;
    end

    function [5:0] map_asp2;
        input signed [7:0] val;
        begin
            case (val)
                -128   : map_asp2 =  6'd7;
                -127   : map_asp2 =  6'd7;
                -126   : map_asp2 = 6'd15;
                -125   : map_asp2 = 6'd15;
                -124   : map_asp2 = 6'd23;
                -123   : map_asp2 = 6'd23;
                -122   : map_asp2 = 6'd23;
                -121   : map_asp2 = 6'd31;
                -120   : map_asp2 = 6'd31;
                -119   : map_asp2 = 6'd31;
                -118   : map_asp2 = 6'd31;
                -117   : map_asp2 = 6'd31;
                -116   : map_asp2 = 6'd31;
                -115   : map_asp2 = 6'd39;
                -114   : map_asp2 = 6'd39;
                -113   : map_asp2 = 6'd39;
                -112   : map_asp2 = 6'd39;
                -111   : map_asp2 = 6'd39;
                -110   : map_asp2 = 6'd39;
                -109   : map_asp2 = 6'd39;
                -108   : map_asp2 = 6'd39;
                -107   : map_asp2 = 6'd39;
                -106   : map_asp2 = 6'd39;
                -105   : map_asp2 = 6'd39;
                -104   : map_asp2 = 6'd39;
                -103   : map_asp2 = 6'd47;
                -102   : map_asp2 = 6'd47;
                -101   : map_asp2 = 6'd47;
                -100   : map_asp2 = 6'd47;
                -99    : map_asp2 = 6'd47;
                -98    : map_asp2 = 6'd47;
                -97    : map_asp2 = 6'd47;
                -96    : map_asp2 = 6'd47;
                -95    : map_asp2 = 6'd47;
                -94    : map_asp2 = 6'd47;
                -93    : map_asp2 = 6'd47;
                -92    : map_asp2 = 6'd47;
                -91    : map_asp2 = 6'd47;
                -90    : map_asp2 = 6'd47;
                -89    : map_asp2 = 6'd47;
                -88    : map_asp2 = 6'd47;
                -87    : map_asp2 = 6'd47;
                -86    : map_asp2 = 6'd47;
                -85    : map_asp2 = 6'd47;
                -84    : map_asp2 = 6'd47;
                -83    : map_asp2 = 6'd47;
                -82    : map_asp2 = 6'd47;
                -81    : map_asp2 = 6'd47;
                -80    : map_asp2 = 6'd47;
                -79    : map_asp2 = 6'd55;
                -78    : map_asp2 = 6'd55;
                -77    : map_asp2 = 6'd55;
                -76    : map_asp2 = 6'd55;
                -75    : map_asp2 = 6'd55;
                -74    : map_asp2 = 6'd55;
                -73    : map_asp2 = 6'd55;
                -72    : map_asp2 = 6'd55;
                -71    : map_asp2 = 6'd55;
                -70    : map_asp2 = 6'd55;
                -69    : map_asp2 = 6'd55;
                -68    : map_asp2 = 6'd55;
                -67    : map_asp2 = 6'd55;
                -66    : map_asp2 = 6'd55;
                -65    : map_asp2 = 6'd55;
                -64    : map_asp2 = 6'd55;
                -63    : map_asp2 =  6'd6;
                -62    : map_asp2 = 6'd14;
                -61    : map_asp2 = 6'd14;
                -60    : map_asp2 = 6'd22;
                -59    : map_asp2 = 6'd22;
                -58    : map_asp2 = 6'd22;
                -57    : map_asp2 = 6'd30;
                -56    : map_asp2 = 6'd30;
                -55    : map_asp2 = 6'd30;
                -54    : map_asp2 = 6'd30;
                -53    : map_asp2 = 6'd30;
                -52    : map_asp2 = 6'd30;
                -51    : map_asp2 = 6'd38;
                -50    : map_asp2 = 6'd38;
                -49    : map_asp2 = 6'd38;
                -48    : map_asp2 = 6'd38;
                -47    : map_asp2 = 6'd38;
                -46    : map_asp2 = 6'd38;
                -45    : map_asp2 = 6'd38;
                -44    : map_asp2 = 6'd38;
                -43    : map_asp2 = 6'd38;
                -42    : map_asp2 = 6'd38;
                -41    : map_asp2 = 6'd38;
                -40    : map_asp2 = 6'd38;
                -39    : map_asp2 = 6'd46;
                -38    : map_asp2 = 6'd46;
                -37    : map_asp2 = 6'd46;
                -36    : map_asp2 = 6'd46;
                -35    : map_asp2 = 6'd46;
                -34    : map_asp2 = 6'd46;
                -33    : map_asp2 = 6'd46;
                -32    : map_asp2 = 6'd46;
                -31    : map_asp2 =  6'd5;
                -30    : map_asp2 = 6'd13;
                -29    : map_asp2 = 6'd13;
                -28    : map_asp2 = 6'd21;
                -27    : map_asp2 = 6'd21;
                -26    : map_asp2 = 6'd21;
                -25    : map_asp2 = 6'd29;
                -24    : map_asp2 = 6'd29;
                -23    : map_asp2 = 6'd29;
                -22    : map_asp2 = 6'd29;
                -21    : map_asp2 = 6'd29;
                -20    : map_asp2 = 6'd29;
                -19    : map_asp2 = 6'd37;
                -18    : map_asp2 = 6'd37;
                -17    : map_asp2 = 6'd37;
                -16    : map_asp2 = 6'd37;
                -15    : map_asp2 =  6'd4;
                -14    : map_asp2 = 6'd12;
                -13    : map_asp2 = 6'd12;
                -12    : map_asp2 = 6'd20;
                -11    : map_asp2 = 6'd20;
                -10    : map_asp2 = 6'd20;
                -9     : map_asp2 = 6'd28;
                -8     : map_asp2 = 6'd28;
                -7     : map_asp2 =  6'd3;
                -6     : map_asp2 = 6'd11;
                -5     : map_asp2 = 6'd11;
                -4     : map_asp2 = 6'd19;
                -3     : map_asp2 =  6'd2;
                -2     : map_asp2 = 6'd10;
                -1     : map_asp2 =  6'd1;
                0      : map_asp2 =  6'd0;
                1      : map_asp2 =  6'd8;
                2      : map_asp2 = 6'd17;
                3      : map_asp2 = 6'd16;
                4      : map_asp2 = 6'd26;
                5      : map_asp2 = 6'd25;
                6      : map_asp2 = 6'd25;
                7      : map_asp2 = 6'd24;
                8      : map_asp2 = 6'd35;
                9      : map_asp2 = 6'd35;
                10     : map_asp2 = 6'd34;
                11     : map_asp2 = 6'd34;
                12     : map_asp2 = 6'd34;
                13     : map_asp2 = 6'd33;
                14     : map_asp2 = 6'd33;
                15     : map_asp2 = 6'd32;
                16     : map_asp2 = 6'd44;
                17     : map_asp2 = 6'd44;
                18     : map_asp2 = 6'd44;
                19     : map_asp2 = 6'd44;
                20     : map_asp2 = 6'd43;
                21     : map_asp2 = 6'd43;
                22     : map_asp2 = 6'd43;
                23     : map_asp2 = 6'd43;
                24     : map_asp2 = 6'd43;
                25     : map_asp2 = 6'd43;
                26     : map_asp2 = 6'd42;
                27     : map_asp2 = 6'd42;
                28     : map_asp2 = 6'd42;
                29     : map_asp2 = 6'd41;
                30     : map_asp2 = 6'd41;
                31     : map_asp2 = 6'd40;
                32     : map_asp2 = 6'd53;
                33     : map_asp2 = 6'd53;
                34     : map_asp2 = 6'd53;
                35     : map_asp2 = 6'd53;
                36     : map_asp2 = 6'd53;
                37     : map_asp2 = 6'd53;
                38     : map_asp2 = 6'd53;
                39     : map_asp2 = 6'd53;
                40     : map_asp2 = 6'd52;
                41     : map_asp2 = 6'd52;
                42     : map_asp2 = 6'd52;
                43     : map_asp2 = 6'd52;
                44     : map_asp2 = 6'd52;
                45     : map_asp2 = 6'd52;
                46     : map_asp2 = 6'd52;
                47     : map_asp2 = 6'd52;
                48     : map_asp2 = 6'd52;
                49     : map_asp2 = 6'd52;
                50     : map_asp2 = 6'd52;
                51     : map_asp2 = 6'd52;
                52     : map_asp2 = 6'd51;
                53     : map_asp2 = 6'd51;
                54     : map_asp2 = 6'd51;
                55     : map_asp2 = 6'd51;
                56     : map_asp2 = 6'd51;
                57     : map_asp2 = 6'd51;
                58     : map_asp2 = 6'd50;
                59     : map_asp2 = 6'd50;
                60     : map_asp2 = 6'd50;
                61     : map_asp2 = 6'd49;
                62     : map_asp2 = 6'd49;
                63     : map_asp2 = 6'd48;
                64     : map_asp2 = 6'd62;
                65     : map_asp2 = 6'd62;
                66     : map_asp2 = 6'd62;
                67     : map_asp2 = 6'd62;
                68     : map_asp2 = 6'd62;
                69     : map_asp2 = 6'd62;
                70     : map_asp2 = 6'd62;
                71     : map_asp2 = 6'd62;
                72     : map_asp2 = 6'd62;
                73     : map_asp2 = 6'd62;
                74     : map_asp2 = 6'd62;
                75     : map_asp2 = 6'd62;
                76     : map_asp2 = 6'd62;
                77     : map_asp2 = 6'd62;
                78     : map_asp2 = 6'd62;
                79     : map_asp2 = 6'd62;
                80     : map_asp2 = 6'd61;
                81     : map_asp2 = 6'd61;
                82     : map_asp2 = 6'd61;
                83     : map_asp2 = 6'd61;
                84     : map_asp2 = 6'd61;
                85     : map_asp2 = 6'd61;
                86     : map_asp2 = 6'd61;
                87     : map_asp2 = 6'd61;
                88     : map_asp2 = 6'd61;
                89     : map_asp2 = 6'd61;
                90     : map_asp2 = 6'd61;
                91     : map_asp2 = 6'd61;
                92     : map_asp2 = 6'd61;
                93     : map_asp2 = 6'd61;
                94     : map_asp2 = 6'd61;
                95     : map_asp2 = 6'd61;
                96     : map_asp2 = 6'd61;
                97     : map_asp2 = 6'd61;
                98     : map_asp2 = 6'd61;
                99     : map_asp2 = 6'd61;
                100    : map_asp2 = 6'd61;
                101    : map_asp2 = 6'd61;
                102    : map_asp2 = 6'd61;
                103    : map_asp2 = 6'd61;
                104    : map_asp2 = 6'd60;
                105    : map_asp2 = 6'd60;
                106    : map_asp2 = 6'd60;
                107    : map_asp2 = 6'd60;
                108    : map_asp2 = 6'd60;
                109    : map_asp2 = 6'd60;
                110    : map_asp2 = 6'd60;
                111    : map_asp2 = 6'd60;
                112    : map_asp2 = 6'd60;
                113    : map_asp2 = 6'd60;
                114    : map_asp2 = 6'd60;
                115    : map_asp2 = 6'd60;
                116    : map_asp2 = 6'd59;
                117    : map_asp2 = 6'd59;
                118    : map_asp2 = 6'd59;
                119    : map_asp2 = 6'd59;
                120    : map_asp2 = 6'd59;
                121    : map_asp2 = 6'd59;
                122    : map_asp2 = 6'd58;
                123    : map_asp2 = 6'd58;
                124    : map_asp2 = 6'd58;
                125    : map_asp2 = 6'd57;
                126    : map_asp2 = 6'd57;
                127    : map_asp2 = 6'd56;
                default: map_asp2 =  6'd0;
            endcase
        end
    endfunction

endmodule