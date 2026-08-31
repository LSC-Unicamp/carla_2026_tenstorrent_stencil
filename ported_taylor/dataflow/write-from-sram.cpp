// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
  // Drains the computed output tiles from cb_in back to DRAM.
  //
  // Runtime arguments:
  // - n_tiles: number of contiguous output tiles assigned to this core
  // - dst_dram_addr: base address of the global output buffer
  // - my_start_tile: first output tile index owned by this core
  // - cb_in: circular buffer containing ready output tiles
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  uint32_t dst_dram_addr = get_arg_val<uint32_t>(1);
  uint32_t my_start_tile = get_arg_val<uint32_t>(2);

  // The circular buffer that we are going to read from and write to DRAM
  uint32_t cb_in = get_arg_val<uint32_t>(3);
  uint32_t tile_size_bytes = get_tile_size(cb_in);

  const InterleavedAddrGenFast<true> out0 = {
      .bank_base_address = dst_dram_addr,   // The base address of the buffer
      .page_size = tile_size_bytes,         // The size of a buffer page
      .data_format = DataFormat::Float16_b, // The data format of the buffer
  };

  // Loop over all the tiles and write them to the output buffer
  for (uint32_t i = 0; i < n_tiles; i++) {

    // write the tile to DRAM
    noc_async_write_tile(my_start_tile + i, out0, get_read_ptr(cb_in));
    noc_async_write_barrier();
    cb_pop_front(cb_in, 1); // increments the read ptr
  }
}
