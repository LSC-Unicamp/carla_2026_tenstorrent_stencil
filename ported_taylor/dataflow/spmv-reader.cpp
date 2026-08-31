// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

#include "send-receive.hpp"

#define have_north(directions) ((directions) & 0x1)
#define have_south(directions) ((directions) & 0x2)
#define have_west(directions) ((directions) & 0x4)
#define have_east(directions) ((directions) & 0x8)

constexpr uint32_t SIX = 0x40c040c0;

// Fills one tile in a circular buffer with a repeated 32-bit pattern.
//
// Intended for scalar/constant tiles used by compute kernels. The value is
// written across the entire tile payload in L1 starting at the CB write pointer
// plus the tile offset.
void fill_tile(uint32_t tile_bytes, uint32_t cb_id, uint32_t tile_id,
               uint32_t val) {

  // Fill 2 uint16 datums in each writes to optimize for performance
  volatile tt_l1_ptr uint32_t *ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(get_write_ptr(cb_id) +
                                                      tile_id * tile_bytes);
  int num_uint32_datums_tile = tile_bytes / 4;
  for (int k = 0; k < num_uint32_datums_tile; k++) {
    ptr[k] = val;
  }
}

void kernel_main() {
  // Reader kernel orchestration for the standalone SpMV example.
  //
  // Responsibilities:
  // - Publish the already-loaded input tiles to the compute kernel.
  // - Materialize the scalar tile containing 6 for the center coefficient.
  // - Build stencil halo components (N/S and E/W) for each local tile.
  // - Coordinate with compute using cb_from_reader tokens and semaphores.
  //
  // Execution order per tile:
  // 1) Wait for compute to stage north/south boundary data.
  // 2) Apply boundary conditions and exchange north/south halos.
  // 3) Push one token to cb_from_reader.
  // 4) Wait for compute to stage west/east boundary data in transposed layout.
  // 5) Apply boundary conditions and exchange west/east halos.
  // 6) Push a second token to cb_from_reader.
  uint32_t n_tiles = get_arg_val<uint32_t>(0);
  uint32_t have_directions = get_arg_val<uint32_t>(1);
  uint32_t north_noc_x = get_arg_val<uint32_t>(2);
  uint32_t north_noc_y = get_arg_val<uint32_t>(3);
  uint32_t south_noc_x = get_arg_val<uint32_t>(4);
  uint32_t south_noc_y = get_arg_val<uint32_t>(5);
  uint32_t west_noc_x = get_arg_val<uint32_t>(6);
  uint32_t west_noc_y = get_arg_val<uint32_t>(7);
  uint32_t east_noc_x = get_arg_val<uint32_t>(8);
  uint32_t east_noc_y = get_arg_val<uint32_t>(9);
  uint32_t A_semaphore_addr = get_semaphore(get_arg_val<uint32_t>(10));
  uint32_t B_semaphore_addr =
      get_semaphore(get_arg_val<uint32_t>(11)); // for north/south halo exchange
  uint32_t A_semaphore_addr_EW = get_semaphore(get_arg_val<uint32_t>(12));
  uint32_t B_semaphore_addr_EW = get_semaphore(get_arg_val<uint32_t>(13));
  uint32_t cb_core = get_arg_val<uint32_t>(14);
  uint32_t cb_im0 = get_arg_val<uint32_t>(15);
  uint32_t cb_core_scale = get_arg_val<uint32_t>(16);
  uint32_t cb_shifted_tiles = get_arg_val<uint32_t>(17);
  uint32_t cb_from_reader = get_arg_val<uint32_t>(18);

  // The compute kernel consumes cb_core directly inside the same program, so
  // publishing n_tiles here makes those staged input tiles visible to it.
  int halo_size = 16;               // number of slots to halo size over
  int offset_bytes = 2 * halo_size; // offset = number of bytes to halo size
  int offset_fp32 = halo_size / 2;  // offset_fp32, for zero filling

  uint32_t tile_size_bytes = get_tile_size(cb_core);

  cb_push_back(cb_core, n_tiles); // reserve output space

  cb_reserve_back(cb_core_scale, 1);
  fill_tile(tile_size_bytes, cb_core_scale, 0, SIX);
  cb_push_back(cb_core_scale, 1);

  for (uint32_t i = 0; i < n_tiles; i++) {
    /************ NORTH/SOUTH CONSTRUCTION ************** */
    cb_wait_front(cb_shifted_tiles, 2);
    uint32_t cb_shifted_tiles_ptr0 = get_read_ptr(cb_shifted_tiles);
    uint32_t cb_shifted_tiles_ptr1 = cb_shifted_tiles_ptr0 + tile_size_bytes;

    // NORTH
    if (!have_north(have_directions)) {
      volatile tt_l1_ptr uint32_t *north_ptr =
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
              cb_shifted_tiles_ptr0);
      for (int i = 0; i < 8; i++)
        north_ptr[i] = 0;
    }

    // SOUTH
    if (!have_south(have_directions)) {
      volatile tt_l1_ptr uint32_t *south_ptr =
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
              cb_shifted_tiles_ptr1);
      for (int i = 0; i < offset_fp32; i++)
        south_ptr[(tile_size_bytes / 4 - 1) - i] = 0;
    }

    uint32_t north_dst = cb_shifted_tiles_ptr1 + tile_size_bytes - offset_bytes;
    uint32_t north_src = cb_shifted_tiles_ptr0 + offset_bytes;
    uint32_t south_src =
        cb_shifted_tiles_ptr1 + tile_size_bytes - offset_bytes * 2;
    halo_exchange(have_north(have_directions), north_noc_x, north_noc_y,
                  have_south(have_directions), south_noc_x, south_noc_y,
                  A_semaphore_addr, B_semaphore_addr,
                  north_dst,             // north dst addr
                  north_src,             // north source
                  cb_shifted_tiles_ptr0, // south dst
                  south_src,             // south source
                  offset_bytes);
    cb_pop_front(cb_shifted_tiles, 2);
    cb_push_back(cb_from_reader, 1);

    cb_wait_front(cb_shifted_tiles, 2);

    uint32_t cb_shifted_tiles_ptr2 = get_read_ptr(cb_shifted_tiles);
    uint32_t cb_shifted_tiles_ptr3 = cb_shifted_tiles_ptr2 + tile_size_bytes;

    // WEST
    if (!have_west(have_directions)) {
      volatile tt_l1_ptr uint32_t *west_ptr =
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
              cb_shifted_tiles_ptr2);
      for (int row = 0; row < 4; row++) {
        int row_offset = row * (halo_size * offset_fp32);
        for (int i = 0; i < offset_fp32; i++) {
          west_ptr[i + row_offset] = 0;
        }
      }
    }

    // EAST
    if (!have_east(have_directions)) {
      volatile tt_l1_ptr uint32_t *east_ptr =
          reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
              cb_shifted_tiles_ptr3);
      for (int row = 0; row < 4; row++) {
        int row_offset = row * (halo_size * offset_fp32) +
                         (halo_size - 1) * ((int(halo_size / 2)));
        for (int i = 0; i < offset_fp32; i++) {
          east_ptr[i + row_offset] = 0;
        }
      }
    }

    // EW Halo exchange.
    // TODO: optimize by putting loop inside of halo exchange

    halo_exchange_transpose(
        have_west(have_directions), west_noc_x, west_noc_y,
        have_east(have_directions), east_noc_x, east_noc_y, A_semaphore_addr_EW,
        B_semaphore_addr_EW, offset_bytes, halo_size, cb_shifted_tiles_ptr2,
        cb_shifted_tiles_ptr3); // row of 16 bfloats,offset_bytes bytes

    cb_pop_front(cb_shifted_tiles, 2);
    /************************** */
    cb_push_back(cb_from_reader,
                 1); // signal that east/west halos are ready
  }
}
