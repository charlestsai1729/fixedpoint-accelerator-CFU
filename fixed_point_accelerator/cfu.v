// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



module Cfu (
    input               cmd_valid,
    output              cmd_ready,
    input      [9:0]    cmd_payload_function_id,
    input      [31:0]   cmd_payload_inputs_0,
    input      [31:0]   cmd_payload_inputs_1,
    output reg          rsp_valid,
    input               rsp_ready,
    output reg [31:0]   rsp_payload_outputs_0,
    input               reset,
    input               clk
);
    // localparam InputOffset = $signed(9'd128);
    reg [8:0] InputOffset;
    // SIMD multiply step:
    wire signed [15:0] prod_0, prod_1, prod_2, prod_3;
    assign prod_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + $signed(InputOffset))
                    * $signed(cmd_payload_inputs_1[7 : 0]);
    assign prod_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + $signed(InputOffset))
                    * $signed(cmd_payload_inputs_1[15: 8]);
    assign prod_2 =  ($signed(cmd_payload_inputs_0[23:16]) + $signed(InputOffset))
                    * $signed(cmd_payload_inputs_1[23:16]);
    assign prod_3 =  ($signed(cmd_payload_inputs_0[31:24]) + $signed(InputOffset))
                    * $signed(cmd_payload_inputs_1[31:24]);
                
    wire signed [31:0] sum_prods;
    assign sum_prods = prod_0 + prod_1 + prod_2 + prod_3;

    assign cmd_ready = ~rsp_valid & ~calculating;

    reg [31:0] exp_input, exp_integer_bits;
    reg exp_input_valid;
    wire exp_finish;
    wire [31:0] exp_result;

    exp EXP (
        .clk(clk),
        .rst(reset),
        .x(exp_input),
        .integer_bits(exp_integer_bits[3:0]),
        .input_valid(exp_input_valid),
        .exp_x(exp_result),
        .output_valid(exp_finish)
    );

    reg [31:0] re_input;
    reg re_input_valid;
    wire [31:0] re_result;
    wire re_finish;

    reciprocal RECIPROCAL (
        .clk(clk),
        .rst(reset),
        .x(re_input),
        .input_valid(re_input_valid),
        .reciprocal(re_result),
        .output_valid(re_finish)
    );

    wire signed [31:0] quantized_result;
    reg [31:0] quantized_multiplier, quantized_shift, quantized_x;
    reg quan_input_valid;
    wire quan_finish;

    MultiplyByQuantizedMultiplier MBQM (
        .clk(clk),
        .x(quantized_x),
        .quantized_multiplier(quantized_multiplier),
        .shift(quantized_shift),
        .result(quantized_result),
        .input_valid(quan_input_valid),
        .output_valid(quan_finish)
    );

    reg calculating;

    always @(posedge clk) begin
        if (reset) begin
            rsp_payload_outputs_0 <= 32'b0;
            rsp_valid <= 1'b0;

            calculating <= 0;

            exp_input <= 0;
            exp_input_valid <= 0;

            re_input <= 0;
            re_input_valid <= 0;
 
            InputOffset <= 0;

        end else if (rsp_valid) begin
            // Waiting to hand off response to CPU.
            rsp_valid <= ~rsp_ready;
        end else if (cmd_valid) begin
            if (cmd_payload_function_id[9:3] == 0) begin
                rsp_payload_outputs_0 <= 32'b0;
                InputOffset <= cmd_payload_inputs_0[8:0];
                rsp_valid <= 1'b1;
            end 
            else if (cmd_payload_function_id[9:3] == 1) begin
                rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
                rsp_valid <= 1'b1;
            end 
            else if (cmd_payload_function_id[9:3] == 10) begin
                quantized_x <= cmd_payload_inputs_0;
                quantized_multiplier <= cmd_payload_inputs_1;

                rsp_valid <= 1'b1;
            end
            else if (cmd_payload_function_id[9:3] == 11) begin
                quantized_shift <= cmd_payload_inputs_0;

                quan_input_valid <= 1;
                calculating <= 1;
            end
            else if (cmd_payload_function_id[9:3] == 12) begin
                exp_input <= cmd_payload_inputs_0;
                exp_integer_bits <= cmd_payload_inputs_1;

                exp_input_valid <= 1;
                calculating <= 1;
            end
            else if (cmd_payload_function_id[9:3] == 13) begin
                re_input <= cmd_payload_inputs_0;

                re_input_valid <= 1;
                calculating <= 1;
            end
        end
        else if (exp_input_valid || re_input_valid || quan_input_valid) begin
            exp_input_valid <= 0;
            re_input_valid <= 0;
            quan_input_valid <= 0;
        end
        else if (exp_finish == 1) begin
            calculating <= 0;
            rsp_valid <= 1'b1;
            rsp_payload_outputs_0 <= exp_result;
        end
        else if (re_finish == 1) begin
            calculating <= 0;
            rsp_valid <= 1'b1;
            rsp_payload_outputs_0 <= re_result;
        end
        else if (quan_finish == 1) begin
            calculating <= 0;
            rsp_valid <= 1'b1;
            rsp_payload_outputs_0 <= quantized_result;
        end
    end
endmodule

module exp(
    input wire clk,
    input wire rst,
    input wire [31:0] x,
    input wire [3:0] integer_bits,
    input wire input_valid,
    output reg [31:0] exp_x,
    output reg output_valid
);

    localparam IDLE = 0, CALC = 1, DONE = 2, INIT = 3;
    reg [1:0] state = IDLE;
    reg [2:0] i = 0;

    reg [36:0] x_extention;
    reg [36:0] temp; // result reg
    reg [73:0] x_power;
    reg [73:0] term;

    reg [15:0] multiplier [0:5];

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            temp <= 37'd0;
            x_power <= 74'd0;
            term <= 74'd0;
            i <= 0;
            x_extention <= 37'd0;
            output_valid <= 0;

            multiplier[0] <= 16'b1000000000000000;
            multiplier[1] <= 16'b0010101010101010;
            multiplier[2] <= 16'b0000101010101010;
            multiplier[3] <= 16'b0000001000100010;
            multiplier[4] <= 0;
            multiplier[5] <= 0;
        end 
        else begin
            case(state)
                IDLE: begin
                    output_valid <= 0;
                    if (input_valid) begin
                        x_extention <= {6'b0, x[30:0]} << integer_bits; // q -> q0
                        temp <= 64'd1 << 31; // initialize temp = 1
                        i <= 1;
                        state <= INIT;
                    end
                end
                INIT: begin
                    x_power <= (x_extention * x_extention) >> 31;
                    term <= x_extention;
                    i <= i + 1;

                    state <= CALC;
                end
                CALC: begin
                    if (i <= 6) begin
                        term <= (x_power * multiplier[i-2]) >> 16;
                        temp <= (i % 2 == 1) ? temp + term : temp - term;
                        x_power <= (x_power * x_extention) >> 31;
                        i <= i + 1;
                    end else begin
                        state <= DONE;
                    end
                end
                DONE: begin
                    state <= IDLE;
                    exp_x <= temp[31] == 1 ? temp[31:0] - 1 : temp[31:0];
                    output_valid <= 1;
                end
            endcase
        end
    end
endmodule

module reciprocal(
    input wire clk,
    input wire rst,
    input wire [31:0] x,
    input wire input_valid,
    output reg [31:0] reciprocal,
    output reg output_valid
);

    // Constants (scaled for your fixed-point representation)
    /////////////////////////////////////////////////////
    // 1 sign bit | 2 integer bits | 31 fractional bit //
    /////////////////////////////////////////////////////
    localparam [33:0] CONSTANT_48_OVER_17 = 34'b0_10_1101001011010010110100101101001; // Approximation of 48/17
    localparam [33:0] CONSTANT_32_OVER_17 = 34'b0_01_1110000111100001111000011110000; // Approximation of 32/17
    localparam IDLE = 0, INIT = 1, CALC1 = 2, CALC2 = 3, CALC3 = 4, CALC4 = 5, DONE = 6;
    reg [2:0] state; 

    // Internal registers for computation
    /////////////////////////////////////////////////////
    // 1 sign bit | 2 integer bits | 31 fractional bit //
    /////////////////////////////////////////////////////
    reg [33:0] half_denominator;
    reg [33:0] xi;
    reg [33:0] half_denominator_times_xi;
    reg [33:0] one_minus_half_denominator_times_xi;
    reg [33:0] correction;

    // temp variable
    wire [67:0] temp0 = CONSTANT_48_OVER_17 - ((half_denominator * CONSTANT_32_OVER_17) >> 31);
    wire [67:0] temp1 = (half_denominator * xi) >> 31;
    wire [67:0] temp2 = (xi * one_minus_half_denominator_times_xi) >> 31;

    integer iteration_count;

    // State machine actions
    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            half_denominator <= 0;
            xi <= 0;
            half_denominator_times_xi <= 0;
            one_minus_half_denominator_times_xi <= 0;
            correction <= 0;
            iteration_count <= 0;
        end
        else begin
            case (state)
            IDLE: begin
                output_valid <= 0;
                reciprocal <= 0;
                if (input_valid) begin
                    state <= INIT;
                end
                half_denominator <= (x + (34'h1 << 31)) >> 1;
            end
            INIT: begin
                // Initialize variables for computation
                xi <= temp0[33:0];
                iteration_count <= 0;
                state <= CALC1;
            end
            CALC1: begin
                // Calculate half_denominator_times_xi
                half_denominator_times_xi <= temp1[33:0];
                state <= CALC2;
            end
            CALC2: begin
                // Calculate one_minus_half_denominator_times_xi
                if (half_denominator_times_xi > (34'h1 << 31)) begin
                    one_minus_half_denominator_times_xi <= half_denominator_times_xi - (34'h1 << 31);
                end else begin
                    one_minus_half_denominator_times_xi <= (34'h1 << 31) - half_denominator_times_xi;
                end
                state <= CALC3;
            end
            CALC3: begin
                // Calculate correction
                correction <= temp2[33:0];
                state <= CALC4;
            end
            CALC4: begin
                // update xi
                if (half_denominator_times_xi > (34'h1 << 31)) begin
                    xi <= xi - correction;
                end else begin
                    xi <= xi + correction;
                end
                
                iteration_count <= iteration_count + 1;
                if (iteration_count >= 3) begin
                    state <= DONE;
                end else begin
                    state <= CALC1;  // Loop back for more iterations
                end
            end
            DONE: begin
                // Finalize the computation
                reciprocal <= {xi[33], xi[31:1]};  // Adjust output formatting
                output_valid <= 1;
                state <= IDLE;
            end
            endcase
        end
    end
endmodule


module MultiplyByQuantizedMultiplier
(
    input clk,
    input reset,
    input [31:0] x,
    input [31:0] quantized_multiplier,
    input [31:0] shift,
    output reg [31:0] result,
    input input_valid,
    output reg output_valid
);
    wire [31:0] left_shifted;
    wire [31:0] right_shift;
    wire [63:0] mul_result;

    assign left_shifted = x << (shift[31] ? 0 : shift);
    assign right_shift = (shift[31] ? -shift : 0);
    assign mul_result = $signed(left_shifted) * $signed(quantized_multiplier);

    reg valid_1;
    reg [63:0] mul_result_reg;
    reg [31:0] left_shifted_reg, right_shift_reg, quantized_multiplier_reg;

    always @(posedge clk) begin
        if (reset) begin
            valid_1 <= 0;
        end
        else if (input_valid) begin
            valid_1 <= 1;

            mul_result_reg <= mul_result;
            left_shifted_reg <= left_shifted;
            right_shift_reg <= right_shift;

            quantized_multiplier_reg <= quantized_multiplier;
        end 
        else begin
            valid_1 <= 0;
        end
    end

    wire [63:0] adjusted_mul_result;
    wire [31:0] nudge;
    wire overflow;
    wire [31:0] mul_high;

    assign overflow = (left_shifted_reg == quantized_multiplier_reg) && (left_shifted_reg == 32'h80000000);
    assign nudge = mul_result_reg[63] ? 32'hbfffffff : 32'h40000000;
    assign adjusted_mul_result = mul_result_reg + nudge;
    assign mul_high = overflow ? 32'h7fffffff : adjusted_mul_result[62:31];

    reg valid_2;
    reg [31:0] right_shift_reg2;
    reg [31:0] mul_high_reg2;

    always @(posedge clk) begin
        if (reset) begin
            valid_2 <= 0;
        end
        else if (valid_1) begin
            valid_2 <= 1;
            right_shift_reg2 <= right_shift_reg;
            mul_high_reg2 <= mul_high;
        end 
        else begin
            valid_2 <= 0;
        end
    end

    wire [31:0] remainder, threshold, mask;
    wire [31:0] x_shifted;

    assign mask = (32'h1 << right_shift_reg2) - 32'h1;
    assign remainder = mul_high_reg2 & mask;
    assign threshold = (mask >> 1) + (mul_high_reg2[31] ? 1 : 0);

    assign x_shifted = $signed(mul_high_reg2) >>> right_shift_reg2;

    reg valid_3;
    reg [31:0] remainder_reg3, threshold_reg3;
    reg [31:0] x_shifted_reg3;

    always @(posedge clk) begin
        if (reset) begin
            valid_3 <= 0;
        end
        else if (valid_2) begin
            valid_3 <= 1;
            remainder_reg3 <= remainder;
            threshold_reg3 <= threshold;
            x_shifted_reg3 <= x_shifted;
        end 
        else begin
            valid_3 <= 0;
        end
    end

    wire [31:0] x_rounded;

    assign x_rounded = x_shifted_reg3 + (remainder_reg3 > threshold_reg3 ? 1 : 0);

    always @(posedge clk) begin
        if (reset) begin
            output_valid <= 0;
        end
        else if (valid_3) begin
            output_valid <= 1;
            result <= x_rounded;
        end 
        else begin
            output_valid <= 0;
        end
    end
endmodule