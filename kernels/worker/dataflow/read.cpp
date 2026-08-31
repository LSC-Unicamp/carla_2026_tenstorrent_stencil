#include <cstdint>
#include "api/debug/dprint.h"  // required in all kernels using DPRINT
#include "api/dataflow/dataflow_api.h"
#include "tools/profiler/kernel_profiler.hpp"

/**
 * Reader description:
 *
 * Reader increments neightbors' semm_reader.
 * Then waits it's own sem_writer to be incremented
 * to push the circular buffer.
 */

// **Runtime arguments**
//
// * in_addr
// * tile_offset
// * n_tiles
// * n_iterations
// * L_dram_addr
// * U_dram_addr
// * LL_dram_addr
// * UU_dram_addr
// * semaphore_reader
// * semaphore_writer
// * my_x           // relativo ao start_coord
// * my_y           // relativo ao start_coord
// * tile_cols
// * tile_rows
// * n_layers
// * phys_x
// * phys_y
//
// **Compiletime arguments**
//
// * cb_in index
// * cb_out index
// * cb_LU index
// * cb_LLUU index
// * cb_top index
// * cb_bottom index
// * cb_left index
// * cb_right index
// * in TensorAccessorArgs
// * L TensorAccessorArgs
// * U TensorAccessorArgs
//
void kernel_main() {
    DPRINT << "inicializando Reader" << ENDL();
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_offset = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);  // More than one not implemented
    const uint32_t n_iterations = get_arg_val<uint32_t>(3);
    const uint32_t L_dram_addr = get_arg_val<uint32_t>(4);
    const uint32_t U_dram_addr = get_arg_val<uint32_t>(5);
    const uint32_t LL_dram_addr = get_arg_val<uint32_t>(6);
    const uint32_t UU_dram_addr = get_arg_val<uint32_t>(7);

    const auto semaphore_reader = get_semaphore(get_arg_val<uint32_t>(8));
    const auto semaphore_reader_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_reader);

    const auto semaphore_writer = get_semaphore(get_arg_val<uint32_t>(9));
    const auto semaphore_writer_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_writer);

    const uint32_t my_x = get_arg_val<uint32_t>(10);
    const uint32_t my_y = get_arg_val<uint32_t>(11);
    const uint32_t tile_cols = get_arg_val<uint32_t>(12);
    const uint32_t tile_rows = get_arg_val<uint32_t>(13);
    const uint32_t input_n_layers = get_arg_val<uint32_t>(14);
    const uint32_t phys_x = get_arg_val<uint32_t>(15);
    const uint32_t phys_y = get_arg_val<uint32_t>(16);

    const bool has_left = (my_x > 0);
    const bool has_right = (my_x < tile_cols - 1);
    const bool has_top = (my_y > 0);
    const bool has_bottom = (my_y < tile_rows - 1);

    const int neighbors_count = has_left + has_right + has_top + has_bottom;

    DPRINT << "my_x: " << my_x << ", my_y: " << my_y << ENDL();
    DPRINT << "phys_x: " << phys_x << ", phys_y: " << phys_y << ENDL();
    DPRINT << "width: " << tile_cols << ", height: " << tile_rows << ENDL();
    DPRINT << "has_left: " << (int)has_left << ", has_right: " << (int)has_right << ", has_top: " << (int)has_top
           << ", has_bottom: " << (int)has_bottom << ENDL();
    DPRINT << "quantidade vizinhos: " << neighbors_count << ENDL();

    // Compiletime args
    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out = get_compile_time_arg_val(1);
    constexpr uint32_t cb_LU = get_compile_time_arg_val(2);
    constexpr uint32_t cb_LLUU = get_compile_time_arg_val(3);

    constexpr uint32_t cb_top = get_compile_time_arg_val(4);
    constexpr uint32_t cb_bottom = get_compile_time_arg_val(5);
    constexpr uint32_t cb_left = get_compile_time_arg_val(6);
    constexpr uint32_t cb_right = get_compile_time_arg_val(7);

    const uint32_t tile_size_bytes = get_tile_size(cb_in);

    constexpr auto in_args = TensorAccessorArgs<8>();
    const auto in = TensorAccessor(in_args, in_addr, tile_size_bytes);

    constexpr auto L_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    const auto L_dram = TensorAccessor(L_args, L_dram_addr, tile_size_bytes);

    constexpr auto U_args = TensorAccessorArgs<L_args.next_compile_time_args_offset()>();
    const auto U_dram = TensorAccessor(U_args, U_dram_addr, tile_size_bytes);

    constexpr auto LL_args = TensorAccessorArgs<U_args.next_compile_time_args_offset()>();
    const auto LL_dram = TensorAccessor(LL_args, LL_dram_addr, tile_size_bytes);

    constexpr auto UU_args = TensorAccessorArgs<LL_args.next_compile_time_args_offset()>();
    const auto UU_dram = TensorAccessor(UU_args, UU_dram_addr, tile_size_bytes);

    // Primeiro vamos copiar o L e o U para a memória na L1
    DPRINT << "Reading LU e LLUU" << ENDL();
    cb_reserve_back(cb_LU, 2);
    cb_reserve_back(cb_LLUU, 2);
    uint32_t LU_ptr = get_write_ptr(cb_LU);
    uint32_t LLUU_ptr = get_write_ptr(cb_LLUU);
    {
        // DeviceZoneScopedN("READER-INIT-LU-LLUU-DRAM-ISSUE");
        noc_async_read_tile(0, L_dram, LU_ptr);
        noc_async_read_tile(0, U_dram, LU_ptr + tile_size_bytes);
        noc_async_read_tile(0, LL_dram, LLUU_ptr);
        noc_async_read_tile(0, UU_dram, LLUU_ptr + tile_size_bytes);
    }

    DPRINT << "Reading CB_in inicial" << ENDL();

    // Reading own tile from DRAM
    cb_reserve_back(cb_in, input_n_layers);
    uint32_t l1_write_addr = get_write_ptr(cb_in);
    {
        // DeviceZoneScopedN("READER-INIT-CB_IN-DRAM-ISSUE");
        for (uint32_t i = 0; i < input_n_layers; i++) {
            DPRINT << "Layer " << i << "; tile_offset = " << tile_offset + i * (tile_cols * tile_rows) << ENDL();
            noc_async_read_tile(tile_offset + i * (tile_cols * tile_rows), in, l1_write_addr);
            l1_write_addr += tile_size_bytes;
        }
    }
    {
        // DeviceZoneScopedN("READER-INIT-DRAM-BARRIER");
        noc_async_read_barrier();
    }
    cb_push_back(cb_LU, 2);
    cb_push_back(cb_LLUU, 2);
    cb_push_back(cb_in, input_n_layers);

    const auto top_semaphore_noc = (has_top) ? get_noc_addr(phys_x, phys_y - 1, semaphore_reader) : 0;
    const auto bottom_semaphore_noc = (has_bottom) ? get_noc_addr(phys_x, phys_y + 1, semaphore_reader) : 0;
    const auto left_semaphore_noc = (has_left) ? get_noc_addr(phys_x - 1, phys_y, semaphore_reader) : 0;
    const auto right_semaphore_noc = (has_right) ? get_noc_addr(phys_x + 1, phys_y, semaphore_reader) : 0;

    DPRINT << "Reader cfg: my=(" << my_x << "," << my_y << ") phys=(" << phys_x << "," << phys_y << ") has TBLR=("
           << (int)has_top << (int)has_bottom << (int)has_left << (int)has_right << ") layers=" << input_n_layers
           << ENDL();

    for (uint32_t i = 0; i < n_iterations; i++) {
        DPRINT << "Reader: iteration " << i << ENDL();

        cb_reserve_back(cb_left, input_n_layers);
        cb_reserve_back(cb_right, input_n_layers);
        cb_reserve_back(cb_top, input_n_layers);
        cb_reserve_back(cb_bottom, input_n_layers);

        DPRINT << "Esperando a tile do vizinho de baixo no endereço " << get_write_ptr(cb_bottom) << ENDL();

        // Telling neighbors we're ready to receive
        if (has_top) {
            noc_semaphore_inc(top_semaphore_noc, 1);
        }
        if (has_bottom) {
            noc_semaphore_inc(bottom_semaphore_noc, 1);
        }
        if (has_left) {
            noc_semaphore_inc(left_semaphore_noc, 1);
        }
        if (has_right) {
            noc_semaphore_inc(right_semaphore_noc, 1);
        }

        noc_async_full_barrier();

        // Wait for neighbors to write their tiles
        DPRINT << "Receiving neighbor tiles" << ENDL();
        noc_semaphore_wait(semaphore_writer_ptr, neighbors_count * (i + 1));
        DPRINT << "Neihgbor tiles received" << ENDL();

        DPRINT << "Reader iter " << i << ": sem_writer reached " << neighbors_count * (i + 1) << ", pushing halo CBs"
               << ENDL();

        cb_push_back(cb_left, input_n_layers);
        cb_push_back(cb_right, input_n_layers);
        cb_push_back(cb_top, input_n_layers);
        cb_push_back(cb_bottom, input_n_layers);
    }

    DPRINT << "Reader: All iterations completed" << ENDL();
}
