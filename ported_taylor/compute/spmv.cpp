// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"

#include "compute_kernel_api/cb_api.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/transpose_wh_dest.h"

#include "tools/profiler/kernel_profiler.hpp"

#include <cstdint>

void add_two_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
  tile_regs_acquire();
  copy_tile_init(cb_a);
  copy_tile(cb_a, /*offset*/ 0, /*register_offset*/ 0);
  copy_tile(cb_b, /*offset*/ 0, /*register_offset*/ 1);

  add_binary_tile_init();
  add_binary_tile(0, 1, 0);

  tile_regs_commit();
  tile_regs_wait();
  cb_reserve_back(cb_out, 1);

  pack_tile(0, cb_out); // c

  tile_regs_release();
  cb_push_back(cb_out, 1);
}

void sub_two_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
  tile_regs_acquire();
  copy_tile_init(cb_a);
  copy_tile(cb_a, /*offset*/ 0, /*register_offset*/ 0);
  copy_tile(cb_b, /*offset*/ 0, /*register_offset*/ 1);

  sub_binary_tile_init();
  sub_binary_tile(0, 1, 0);

  tile_regs_commit();
  tile_regs_wait();
  cb_reserve_back(cb_out, 1);

  pack_tile(0, cb_out); // c

  tile_regs_release();
  cb_push_back(cb_out, 1);
}

void mul_two_tiles(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out) {
  tile_regs_acquire();
  copy_tile_init(cb_a);
  copy_tile(cb_a, /*offset*/ 0, /*register_offset*/ 0);
  copy_tile(cb_b, /*offset*/ 0, /*register_offset*/ 1);

  mul_binary_tile_init();
  mul_binary_tile(0, 1, 0);

  tile_regs_commit();
  tile_regs_wait();
  cb_reserve_back(cb_out, 1);

  pack_tile(0, cb_out); // c

  tile_regs_release();
  cb_push_back(cb_out, 1);
}

void transpose_one_tile(uint32_t cb_src, uint32_t cb_dst) {
  tile_regs_acquire();
  copy_tile_init(cb_src);
  copy_tile(cb_src, 0, 0);

  transpose_wh_dest_init_short();
  transpose_wh_dest(0);
  tile_regs_commit();

  tile_regs_wait();

  cb_reserve_back(cb_dst, 1);
  pack_tile(0, cb_dst);

  tile_regs_release();

  cb_push_back(cb_dst, 1);
}

void copy_one_tile(uint32_t cb_src, uint32_t cb_dst) {
  tile_regs_acquire();
  copy_tile_init(cb_src);
  copy_tile(cb_src, /*offset*/ 0, /*register_offset*/ 0);

  tile_regs_commit();
  tile_regs_wait();
  cb_reserve_back(cb_dst, 1);

  pack_tile(0, cb_dst); // c

  tile_regs_release();
  cb_push_back(cb_dst, 1);
}

namespace NAMESPACE {
void MAIN {
  // Runtime arguments.
  // - n_tiles: number of tiles owned by this core along the z dimension
  // - cb_core .. cb_out: circular-buffer IDs used for the stencil pipeline
  uint32_t n_tiles = get_arg_val<uint32_t>(0);

  uint32_t cb_core = get_arg_val<uint32_t>(1);
  uint32_t cb_im0 = get_arg_val<uint32_t>(2);
  uint32_t cb_core_scale = get_arg_val<uint32_t>(3);
  uint32_t cb_shifted_tiles = get_arg_val<uint32_t>(4);
  uint32_t cb_from_reader = get_arg_val<uint32_t>(5);
  uint32_t cb_im1 = get_arg_val<uint32_t>(6);

  uint32_t cb_out = get_arg_val<uint32_t>(7);

  cb_wait_front(cb_core, n_tiles);
  cb_wait_front(cb_core_scale, 1);

  // Configure the unary/binary tile operators used below.
  unary_op_init_common(cb_core, cb_out);
  for (uint32_t i = 0; i < n_tiles; i++) {
    DeviceZoneScopedN("Stencil");

    // Stage north and south neighbors into cb_shifted_tiles. The reader kernel
    // later patches boundary regions and exchanges remote halo data in-place.
    // north
    cb_set_read_ptr(cb_core, cb_core, -2);
    copy_one_tile(cb_core, cb_shifted_tiles);
    cb_set_read_ptr(cb_core, cb_core, 2);

    // // // south
    cb_set_read_ptr(cb_core, cb_core, 2);
    copy_one_tile(cb_core, cb_shifted_tiles);
    cb_set_read_ptr(cb_core, cb_core, -2);

    // EW
    transpose_one_tile(cb_core, cb_im0);
    cb_wait_front(cb_im0, 1);
    // west
    cb_set_read_ptr(cb_im0, cb_im0, -2);
    copy_one_tile(cb_im0, cb_shifted_tiles);
    cb_set_read_ptr(cb_im0, cb_im0, 2);

    // east
    cb_set_read_ptr(cb_im0, cb_im0, 2);
    copy_one_tile(cb_im0, cb_shifted_tiles);
    cb_set_read_ptr(cb_im0, cb_im0, -2);

    cb_pop_front(cb_im0, 1);

    // Reader-side barrier token: north/south halos are ready to consume.
    cb_wait_front(cb_from_reader, 1);
    cb_pop_front(cb_from_reader, 1);

    // Start the accumulation with north - 6*x so the center contribution enters
    // with the correct sign.
    // scale middle tile by 6
    mul_two_tiles(cb_core, cb_core_scale, cb_im0);
    cb_wait_front(cb_im0, 1);

    // add north component
    sub_two_tiles(cb_shifted_tiles, cb_im0, cb_im0);
    cb_pop_front(cb_im0, 1);
    cb_set_read_ptr(cb_shifted_tiles, cb_shifted_tiles, 128);

    cb_wait_front(cb_im0, 1);

    //   add south component
    add_two_tiles(cb_shifted_tiles, cb_im0,
                  cb_im1); // todo: change when EW is added
    cb_set_read_ptr(cb_shifted_tiles, cb_shifted_tiles, 128);
    cb_pop_front(cb_im0, 1);

    // Reader-side barrier token: west/east halos are ready to consume.
    cb_wait_front(cb_from_reader, 1);
    cb_pop_front(cb_from_reader, 1);

    // transpose EW back and add west
    transpose_one_tile(cb_shifted_tiles, cb_im0);
    cb_set_read_ptr(cb_shifted_tiles, cb_shifted_tiles, 128);

    cb_wait_front(cb_im0, 1);
    cb_wait_front(cb_im1, 1);
    add_two_tiles(cb_im1, cb_im0, cb_im1);
    cb_wait_front(cb_im1, 1);
    cb_pop_front(cb_im0, 1);
    cb_pop_front(cb_im1, 1);

    // transpose EW back and add east
    transpose_one_tile(cb_shifted_tiles, cb_im0);
    cb_set_read_ptr(cb_shifted_tiles, cb_shifted_tiles, 128);

    cb_set_read_ptr(cb_shifted_tiles, cb_shifted_tiles,
                    -128 * 4); // reset read ptr of shifted tiles

    cb_wait_front(cb_im0, 1);
    add_two_tiles(cb_im1, cb_im0, cb_im1);

    cb_pop_front(cb_im1, 1);
    cb_pop_front(cb_im0, 1);

    if (i != 0) {
      // add previous tile
      int previous_offset = 128;
      cb_wait_front(cb_im1, 1);
      cb_set_read_ptr(cb_core, cb_core, -previous_offset);
      add_two_tiles(cb_core, cb_im1, cb_im1);
      cb_pop_front(cb_im1, 1);
      cb_set_read_ptr(cb_core, cb_core, previous_offset);
    }

    if (i != n_tiles - 1) {
      // add next tile
      int next_offset = 128;
      cb_wait_front(cb_im1, 1);
      cb_set_read_ptr(cb_core, cb_core, next_offset);
      add_two_tiles(cb_core, cb_im1, cb_im1);
      cb_pop_front(cb_im1, 1);
      cb_set_read_ptr(cb_core, cb_core, -next_offset);
    }

    cb_wait_front(cb_im1, 1);
    copy_one_tile(cb_im1, cb_out);
    cb_pop_front(cb_im1, 1);

    cb_pop_front(cb_core, 1); // and this;
  } // namespace NAMESPACE

} // namespace NAMESPACE
} // namespace NAMESPACE
