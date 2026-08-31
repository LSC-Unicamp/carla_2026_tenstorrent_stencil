// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
  // Stages this core's input tiles from DRAM into cb_in.
  //
  // Runtime arguments:
  // - n_tiles: number of contiguous tiles assigned to this core
  // - src_dram_addr: base address of the global input buffer
  // - my_start_tile: first tile index owned by this core
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  uint32_t src_dram_addr = get_arg_val<uint32_t>(1);
  uint32_t my_start_tile = get_arg_val<uint32_t>(2);

  uint32_t cb_in = tt::CBIndex::c_0;
  uint32_t tile_size_bytes = get_tile_size(cb_in);

  const InterleavedAddrGenFast<true> in0 = {
      .bank_base_address = src_dram_addr,   // The base address of the buffer
      .page_size = tile_size_bytes,         // The size of a buffer page
      .data_format = DataFormat::Float16_b, // The data format of the buffer
  };

  for (uint32_t i = 0; i < n_tiles; i++) {
    noc_async_read_tile(my_start_tile + i, in0, get_write_ptr(cb_in));
    noc_async_read_barrier();

    cb_push_back(cb_in, 1); // increments the write ptr
  }
}
