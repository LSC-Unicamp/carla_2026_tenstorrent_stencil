#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/fill.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/cb_api.h"
#include "tools/profiler/kernel_profiler.hpp"

// **Runtime arguments**
//
// * n_tiles
// * n_iterations
// * my_x
// * my_y
// * tile_cols
// * tile_rows
// * input_n_layers
//
// ** Compiletime arguments**
//
// * cb_in1 index
// * cb_in2 index
// * cb_out index
// * cb_LU index
// * cb_LLUU index
// * cb_top index
// * cb_bottom index
// * cb_left index
// * cb_right index
// * cb_zero index

void kernel_main() {
    DPRINT_MATH(DPRINT << "norm scalar bits=0x3e2aaaab (expected ~1/6)" << ENDL(););

    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t n_iterations = get_arg_val<uint32_t>(1);
    const uint32_t my_x = get_arg_val<uint32_t>(2);
    const uint32_t my_y = get_arg_val<uint32_t>(3);
    const uint32_t tile_cols = get_arg_val<uint32_t>(4);
    const uint32_t tile_rows = get_arg_val<uint32_t>(5);
    const uint32_t input_n_layers = get_arg_val<uint32_t>(6);

    const bool has_left = (my_x > 0);
    const bool has_right = (my_x < tile_cols - 1);
    const bool has_top = (my_y > 0);
    const bool has_bottom = (my_y < tile_rows - 1);

    // Compiletime args
    constexpr uint32_t cb_in1 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_in2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    // L primeiro, U depois
    constexpr uint32_t cb_LU = get_compile_time_arg_val(3);
    // LL primeiro, UU depois
    constexpr uint32_t cb_LLUU = get_compile_time_arg_val(4);

    constexpr uint32_t cb_top = get_compile_time_arg_val(5);
    constexpr uint32_t cb_bottom = get_compile_time_arg_val(6);
    constexpr uint32_t cb_left = get_compile_time_arg_val(7);
    constexpr uint32_t cb_right = get_compile_time_arg_val(8);
    constexpr uint32_t cb_zero = get_compile_time_arg_val(9);

    constexpr uint32_t dst_reg = 0;

    // Usamos dois cb_in para fazer o ping pong a cada iteração
    uint32_t cb_in = cb_in1;

    mm_init(cb_in, cb_LU, cb_out);

    // Filling cb_zero with zeros
    ckernel::tile_regs_acquire();

    fill_tile_init();
    fill_tile_int(dst_reg, 0);

    ckernel::tile_regs_commit();

    ckernel::cb_reserve_back(cb_zero, 1);

    ckernel::tile_regs_wait();

    ckernel::pack_tile(dst_reg, cb_zero);

    ckernel::tile_regs_release();

    ckernel::cb_push_back(cb_zero, 1);

    // Waiting all tile be available
    // Esperamos todas as tiles estarem disponíveis
    cb_wait_front(cb_LU, 2);
    cb_wait_front(cb_LLUU, 2);

    // Copying CB to writer before first iteration
    copy_tile_init(cb_in);

    for (uint32_t i = 0; i < input_n_layers; i++) {
        cb_wait_front(cb_in, 1);
        ckernel::tile_regs_acquire();

        ckernel::copy_tile(cb_in, i, dst_reg);

        ckernel::tile_regs_commit();

        ckernel::cb_reserve_back(cb_out, 1);

        ckernel::tile_regs_wait();

        ckernel::pack_tile(dst_reg, cb_out);

        ckernel::tile_regs_release();

        ckernel::cb_push_back(cb_out, 1);
    }

    uint32_t cb_next;
    for (uint32_t i = 0; i < n_iterations; i++) {
        // DeviceZoneScopedN("Stencil iter");
        // Se o i for par, então cb_in é o cb_in1 e cb_next é o cb_in2
        if (i % 2 == 0) {
            cb_in = cb_in1;
            cb_next = cb_in2;
        } else {
            cb_in = cb_in2;
            cb_next = cb_in1;
        }

        DPRINT_MATH(DPRINT << "Compute iter " << i << ENDL(););
        DPRINT_UNPACK(DPRINT << "Waiting neighbors" << ENDL());

        // We still wait them because, even if we don't need them, reader
        // still pushes them emtpy
        cb_wait_front(cb_in, input_n_layers);
        cb_wait_front(cb_left, input_n_layers);
        cb_wait_front(cb_right, input_n_layers);
        cb_wait_front(cb_top, input_n_layers);
        cb_wait_front(cb_bottom, input_n_layers);

        {
            DeviceZoneScopedN("Stencil");
            for (uint32_t layer = 0; layer < input_n_layers; layer++) {
                tile_regs_acquire();
                // Clearing register
                {
                    MATH(DPRINT << "limpando registrador" << ENDL());
                    fill_tile_init();
                    fill_tile_int(dst_reg, 0);
                }

                // Shift L
                {
                    DPRINT_MATH(DPRINT << "Shift L" << ENDL());
                    ckernel::mm_init(cb_in, cb_LU, cb_out);

                    matmul_tiles(cb_in, cb_LU, layer, 0, dst_reg);  // Matriz esquerda
                    matmul_tiles(cb_LU, cb_in, 0, layer, dst_reg);  // Matriz abaixo
                }

                // Shift U
                {
                    DPRINT_MATH(DPRINT << "Shift U" << ENDL(););
                    mm_init_short(cb_LU, cb_in);

                    matmul_tiles(cb_in, cb_LU, layer, 1, dst_reg);  // Matriz direita
                    matmul_tiles(cb_LU, cb_in, 1, layer, dst_reg);  // Matriz acima
                }

                // If we have a top neighbor, shift it and accumulate
                if (has_top) {
                    DPRINT_MATH(DPRINT << "Top Halo" << ENDL(););
                    mm_init_short(cb_LLUU, cb_top);

                    matmul_tiles(cb_LLUU, cb_top, 1, layer, dst_reg);
                }

                // If we have a bottom neighbor, shift it and accumulate
                if (has_bottom) {
                    DPRINT_MATH(DPRINT << "Bottom Halo" << ENDL(););
                    mm_init_short(cb_LLUU, cb_bottom);

                    matmul_tiles(cb_LLUU, cb_bottom, 0, layer, dst_reg);
                }

                // If we have a left neighbor, shift it and accumulate
                if (has_left) {
                    DPRINT_MATH(DPRINT << "Left Halo" << ENDL(););
                    mm_init_short(cb_left, cb_LLUU);

                    matmul_tiles(cb_left, cb_LLUU, layer, 0, dst_reg);
                }

                // If we have a right neighbor, shift it and accumulate
                if (has_right) {
                    DPRINT_MATH(DPRINT << "Right Halo" << ENDL(););
                    mm_init_short(cb_right, cb_LLUU);

                    matmul_tiles(cb_right, cb_LLUU, layer, 1, dst_reg);
                }

                bool has_above = layer < input_n_layers - 1;
                bool has_below = layer > 0;

                // Top and bottom layers
                {
                    // If has both, sum the top with the bottom
                    if (has_above && has_below) {
                        ckernel::add_tiles_init(cb_in, cb_in, /*acc_to_dst =*/true);
                        DPRINT_MATH(DPRINT << "Summing above and beloy layers" << ENDL());

                        ckernel::add_tiles(cb_in, cb_in, layer - 1, layer + 1, dst_reg);
                    }

                    // If only has one of them, sum that with zero
                    else if (has_above) {
                        ckernel::add_tiles_init(cb_in, cb_zero, /*acc_to_dst =*/true);
                        DPRINT_MATH(DPRINT << "Summing above layer" << ENDL());

                        ckernel::add_tiles(cb_zero, cb_in, 0, layer + 1, dst_reg);
                    }

                    // If we have below layer, sum it
                    else if (has_below) {
                        ckernel::add_tiles_init(cb_in, cb_zero, /*acc_to_dst =*/true);
                        DPRINT_MATH(DPRINT << "Summing below layer" << ENDL());

                        ckernel::add_tiles(cb_zero, cb_in, 0, layer - 1, dst_reg);
                    }
                }

                // Divide result by 6
                {
                    DPRINT_MATH(DPRINT << "Dividing by 6" << ENDL(););
                    binop_with_scalar_tile_init();
                    div_unary_tile(dst_reg, 0x3e2aaaab /*0.6666666666*/);
                    // mul_unary_tile(dst_reg, 0x3e2aaaab /*0.6666666666*/);
                }
                tile_regs_commit();

                cb_reserve_back(cb_out, 1);
                cb_reserve_back(cb_next, 1);
                tile_regs_wait();

                pack_tile(dst_reg, cb_out);
                ckernel::pack_reconfig_data_format(cb_out, cb_next);
                pack_tile(dst_reg, cb_next);
                DPRINT_PACK(DPRINT << "Sending layer " << layer << " to CB_out and CB_in" << ENDL());
                tile_regs_release();
                cb_push_back(cb_next, 1);
                cb_push_back(cb_out, 1);
            }

            cb_pop_front(cb_top, input_n_layers);
            cb_pop_front(cb_bottom, input_n_layers);
            cb_pop_front(cb_left, input_n_layers);
            cb_pop_front(cb_right, input_n_layers);
            cb_pop_front(cb_in, input_n_layers);
        }

        DPRINT_MATH(DPRINT << "Popping CBs" << ENDL());
    }
    DPRINT << "All iterations completed" << ENDL();
}
