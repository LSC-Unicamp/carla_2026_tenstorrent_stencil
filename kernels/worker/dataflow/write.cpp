#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"  // required in all kernels using DPRINT

/**
 * Writer description:
 *
 * Writer waits neighbors increment his sem_reader.
 *
 * Then after sem_reader reaches a value, he sends his tile to
 * the neighbors, then increments their sem_writer to
 * warn new tiles have arrived
 */

// **Runtime arguments**
//
// * in_addr
// * tile_offset
// * n_tiles
// * n_iterations
// * semaphore_reader
// * semaphore_writer
// * my_x
// * my_y
// * tile_cols
// * tile_rows
// * input_n_layers
// * phys_x
// * phys_y
//
// **Compiletime arguments**
//
// * cb_out
// * cb_top
// * cb_bottom
// * cb_left
// * cb_right
// * inA TensorAccessorArgs
void kernel_main() {
    const uint32_t in_addr = get_arg_val<uint32_t>(0);
    const uint32_t tile_offset = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles = get_arg_val<uint32_t>(2);
    const uint32_t n_iterations = get_arg_val<uint32_t>(3);

    const auto semaphore_reader = get_semaphore(get_arg_val<uint32_t>(4));
    const auto semaphore_reader_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_reader);

    const auto semaphore_writer = get_semaphore(get_arg_val<uint32_t>(5));

    const uint32_t my_x = get_arg_val<uint32_t>(6);
    const uint32_t my_y = get_arg_val<uint32_t>(7);

    const uint32_t tile_cols = get_arg_val<uint32_t>(8);
    const uint32_t tile_rows = get_arg_val<uint32_t>(9);
    const uint32_t input_n_layers = get_arg_val<uint32_t>(10);
    const uint32_t phys_x = get_arg_val<uint32_t>(11);
    const uint32_t phys_y = get_arg_val<uint32_t>(12);

    // Compiletime args
    constexpr uint32_t cb_out = get_compile_time_arg_val(0);

    constexpr uint32_t cb_top = get_compile_time_arg_val(1);
    constexpr uint32_t cb_bottom = get_compile_time_arg_val(2);
    constexpr uint32_t cb_left = get_compile_time_arg_val(3);
    constexpr uint32_t cb_right = get_compile_time_arg_val(4);

    const uint32_t tile_size_bytes = get_tile_size(cb_out);

    constexpr auto in_args = TensorAccessorArgs<5>();
    const auto in = TensorAccessor(in_args, in_addr, tile_size_bytes);

    const bool has_left = (my_x > 0);
    const bool has_right = (my_x < tile_cols - 1);
    const bool has_top = (my_y > 0);
    const bool has_bottom = (my_y < tile_rows - 1);

    const int neighbor_count = has_left + has_right + has_top + has_bottom;

    const auto top_semaphore_noc = (has_top) ? get_noc_addr(phys_x, phys_y - 1, semaphore_writer) : 0;
    const auto bottom_semaphore_noc = (has_bottom) ? get_noc_addr(phys_x, phys_y + 1, semaphore_writer) : 0;
    const auto left_semaphore_noc = (has_left) ? get_noc_addr(phys_x - 1, phys_y, semaphore_writer) : 0;
    const auto right_semaphore_noc = (has_right) ? get_noc_addr(phys_x + 1, phys_y, semaphore_writer) : 0;

    for (uint32_t i = 0; i < n_iterations; i++) {
        // DPRINT << "Writer iter " << i << ": my=(" << my_x << "," << my_y << ") phys=(" << phys_x << "," << phys_y
        //        << "), wait target=" << neighbor_count * (i + 1) << ENDL();

        cb_wait_front(cb_out, input_n_layers);

        auto cb_out_ptr = get_read_ptr(cb_out);

        DPRINT << "Waiting neighbors get ready to receive" << ENDL();
        noc_semaphore_wait(semaphore_reader_ptr, neighbor_count * (i + 1));

        {
            // DeviceZoneScopedN("SEND-TO-NEIGHBOR");
            DPRINT << "Sending tiles to neighbors" << ENDL();

            if (has_top) {
                auto cb_bottom_ptr = get_write_ptr(cb_bottom);
                const uint64_t top_core_noc_addr = get_noc_addr(phys_x, phys_y - 1, cb_bottom_ptr);
                noc_async_write(cb_out_ptr, top_core_noc_addr, tile_size_bytes * input_n_layers);
            }
            if (has_bottom) {
                auto cb_top_ptr = get_write_ptr(cb_top);
                const uint64_t bottom_core_noc_addr = get_noc_addr(phys_x, phys_y + 1, cb_top_ptr);
                noc_async_write(cb_out_ptr, bottom_core_noc_addr, tile_size_bytes * input_n_layers);
            }
            if (has_left) {
                auto cb_right_ptr = get_write_ptr(cb_right);
                const uint64_t left_core_noc_addr = get_noc_addr(phys_x - 1, phys_y, cb_right_ptr);
                noc_async_write(cb_out_ptr, left_core_noc_addr, tile_size_bytes * input_n_layers);
            }
            if (has_right) {
                auto cb_left_ptr = get_write_ptr(cb_left);
                const uint64_t right_core_noc_addr = get_noc_addr(phys_x + 1, phys_y, cb_left_ptr);
                noc_async_write(cb_out_ptr, right_core_noc_addr, tile_size_bytes * input_n_layers);
            }

            noc_async_write_barrier();
            cb_pop_front(cb_out, input_n_layers);

            if (has_top) {
                DPRINT << "Incrementing top writer semaphore" << ENDL();
                noc_semaphore_inc(top_semaphore_noc, 1);
            }
            if (has_bottom) {
                DPRINT << "Incrementing bottom writer semaphore" << ENDL();
                noc_semaphore_inc(bottom_semaphore_noc, 1);
            }
            if (has_left) {
                DPRINT << "Incrementing left writer semaphore" << ENDL();
                noc_semaphore_inc(left_semaphore_noc, 1);
            }
            if (has_right) {
                DPRINT << "Incrementing right writer semaphore" << ENDL();
                noc_semaphore_inc(right_semaphore_noc, 1);
            }
        }

        noc_async_full_barrier();
    }

    // Envia o último valor de volta para a DRAM
    DPRINT << "Waiting last cb_out" << ENDL();
    cb_wait_front(cb_out, input_n_layers);
    const uint32_t cb_out_ptr = get_read_ptr(cb_out);
    for (uint32_t layer = 0; layer < input_n_layers; layer++) {
        noc_async_write_tile(tile_offset + layer * (tile_cols * tile_rows), in, cb_out_ptr + layer * tile_size_bytes);
    }
    noc_async_write_barrier();
    cb_pop_front(cb_out, input_n_layers);
    DPRINT << "Tile sent to DRAM" << ENDL();
}
