
#include "api/dataflow/dataflow_api.h"
#include "api/debug/pause.h"
#include "tools/profiler/kernel_profiler.hpp"
#include <cstdint>
#include <stdint.h>

// Signals the remote sender that this core is ready to receive.
//
// This increments the peer's semaphore ("B" in this file's handshake naming),
// allowing the peer-side send path to proceed.
void noc_post_receive(uint64_t src_semaphore_noc_addr) {
  noc_semaphore_inc(src_semaphore_noc_addr, 1); // B = 1
}

// Blocks until incoming data arrival is acknowledged on this core.
//
// Waits until local semaphore A reaches 1, then resets it to 0 so the same
// semaphore can be reused by the next exchange round.
void noc_wait_receive(volatile tt_l1_ptr uint32_t *my_semaphore_ptr) {
  noc_semaphore_wait(my_semaphore_ptr, 1); // wait A = 1
  noc_semaphore_set(my_semaphore_ptr, 0);  // reset for next round
}

// Performs a point-to-point NoC send with semaphore-based synchronization.
//
// Protocol:
// 1) wait for local "ready-to-send" semaphore B,
// 2) write payload from local L1 to remote L1,
// 3) signal remote "data-arrived" semaphore A.
void noc_send(uint64_t dst_semaphore_noc_addr,
              volatile tt_l1_ptr uint32_t *my_semaphore_ptr,
              uint64_t dst_noc_addr, uint32_t my_src_l1_addr,
              uint32_t size_bytes) {

  noc_semaphore_wait(my_semaphore_ptr, 1); // wait B = 1
  noc_semaphore_set(my_semaphore_ptr, 0);  // reset for next round

  // send the data to the neighbor
  noc_async_write(my_src_l1_addr, dst_noc_addr, size_bytes);
  noc_async_write_barrier();

  noc_semaphore_inc(dst_semaphore_noc_addr, 1); // A = 1
}

// Receives data using the paired send/receive semaphore protocol.
//
// First notifies peer readiness, then waits for peer completion signal. This
// helper bundles noc_post_receive + noc_wait_receive for common receive paths.
void noc_receive(volatile tt_l1_ptr uint32_t *my_semaphore_ptr,
                 uint64_t src_semaphore_noc_addr) {
  noc_post_receive(src_semaphore_noc_addr);
  noc_wait_receive(my_semaphore_ptr);
}

// Sends a payload to another core using a destination circular buffer (CB).
//
// Resolves remote write pointer for the selected CB, builds destination NoC
// addresses (data + semaphore), then delegates transfer/handshake to noc_send.
void noc_send_cb(uint32_t my_src_l1_addr, const uint32_t cb, uint32_t noc_x,
                 uint32_t noc_y, uint32_t dst_semaphore_addr,
                 uint32_t src_semaphore_addr, uint32_t size_bytes) {
  uint32_t dst_addr = get_write_ptr(cb);
  uint64_t dst_noc_addr = get_noc_addr(noc_x, noc_y, dst_addr);

  volatile tt_l1_ptr uint32_t *src_semaphore_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(src_semaphore_addr);

  uint64_t dst_semaphore_noc_addr =
      get_noc_addr(noc_x, noc_y, dst_semaphore_addr);

  noc_send(dst_semaphore_noc_addr, src_semaphore_ptr, dst_noc_addr,
           my_src_l1_addr, size_bytes);
}

// Completes a CB-oriented receive from a remote sender.
//
// Computes remote semaphore address from NoC coordinates and waits for the
// sender to finish writing into this core's CB write location.
void noc_receive_cb(uint32_t cb, uint32_t dst_semaphore_addr,
                    uint32_t src_semaphore_addr, uint32_t noc_x,
                    uint32_t noc_y) {
  uint64_t src_semaphore_noc_addr =
      get_noc_addr(noc_x, noc_y, src_semaphore_addr);

  volatile tt_l1_ptr uint32_t *dst_semaphore_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dst_semaphore_addr);
  DPRINT_DATA1(DPRINT << "Receiving bytes at: " << get_write_ptr(cb) << ENDL());
  noc_receive(dst_semaphore_ptr, src_semaphore_noc_addr);
}

// Posts only the "ready-to-receive" side of a CB receive handshake.
//
// Use when posting receive readiness and waiting are split across phases for
// better overlap/pipelining.
void noc_post_receive_cb(uint32_t cb, uint32_t dst_semaphore_addr,
                         uint32_t src_semaphore_addr, uint32_t noc_x,
                         uint32_t noc_y) {
  uint64_t src_semaphore_noc_addr =
      get_noc_addr(noc_x, noc_y, src_semaphore_addr);

  volatile tt_l1_ptr uint32_t *dst_semaphore_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dst_semaphore_addr);
  DPRINT_DATA1(DPRINT << "Receiving bytes at: " << get_write_ptr(cb) << ENDL());
  noc_post_receive(src_semaphore_noc_addr);
}

// Waits only for completion of a previously posted CB receive handshake.
//
// Pairs with noc_post_receive_cb in split-phase communication patterns.
void noc_wait_receive_cb(uint32_t cb, uint32_t dst_semaphore_addr,
                         uint32_t src_semaphore_addr, uint32_t noc_x,
                         uint32_t noc_y) {
  uint64_t src_semaphore_noc_addr =
      get_noc_addr(noc_x, noc_y, src_semaphore_addr);

  volatile tt_l1_ptr uint32_t *dst_semaphore_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(dst_semaphore_addr);
  DPRINT_DATA1(DPRINT << "Receiving bytes at: " << get_write_ptr(cb) << ENDL());
  noc_wait_receive(dst_semaphore_ptr);
}

// Exchanges halo slices with left/right neighbors without transpose.
//
// Behavior by direction:
// - Sends local right-edge data to left neighbor (if present).
// - Receives corresponding halo from right neighbor (if present).
// - Sends local left-edge data to right neighbor (if present).
// - Receives corresponding halo from left neighbor (if present).
//
// Uses two semaphores (A/B) to coordinate readiness and completion on both
// directions, and resets semaphores so the function can be called repeatedly.
void halo_exchange(const uint32_t have_left, uint32_t left_noc_x,
                   uint32_t left_noc_y, const uint32_t have_right,
                   uint32_t right_noc_x, const uint32_t right_noc_y,
                   uint32_t A_semaphore_addr, uint32_t B_semaphore_addr,
                   uint32_t left_dst_addr,
                   uint32_t left_src_l1_addr, // for sending left
                   uint32_t right_dst_addr,
                   uint32_t right_src_l1_addr, // for sending right
                   uint32_t size_bytes) {

  volatile tt_l1_ptr uint32_t *A_semaphore_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(A_semaphore_addr);
  volatile tt_l1_ptr uint32_t *B_semaphore_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(B_semaphore_addr);

  uint64_t leftA_semaphore_noc_addr;

  // send leftwards and receive from right
  uint64_t rightB_semaphore_noc_addr;
  if (have_right) {
    rightB_semaphore_noc_addr =
        get_noc_addr(right_noc_x, right_noc_y, B_semaphore_addr);
    noc_semaphore_inc(rightB_semaphore_noc_addr,
                      1); // ready to receive from right
  }
  if (have_left) {
    leftA_semaphore_noc_addr =
        get_noc_addr(left_noc_x, left_noc_y, A_semaphore_addr);

    uint64_t left_dst_noc_addr =
        get_noc_addr(left_noc_x, left_noc_y, left_dst_addr);
    noc_send(leftA_semaphore_noc_addr,
             B_semaphore_ptr, // my pointer
             left_dst_noc_addr, left_src_l1_addr, size_bytes);
  }

  if (have_right) {
    DPRINT_DATA1(DPRINT << "Waiting receive from right: " << A_semaphore_addr
                        << ENDL());
    noc_semaphore_wait(A_semaphore_ptr, 1); // wait for send to complete
    noc_semaphore_set(A_semaphore_ptr, 0);  // reset A

    rightB_semaphore_noc_addr =
        get_noc_addr(right_noc_x, right_noc_y, B_semaphore_addr);

    DPRINT_DATA1(DPRINT << "Sending right: waiting at A: " << A_semaphore_addr
                        << " and alerting at A: " << A_semaphore_addr
                        << ENDL());
    uint64_t right_dst_noc_addr =
        get_noc_addr(right_noc_x, right_noc_y, right_dst_addr);

    noc_async_write(right_src_l1_addr, right_dst_noc_addr, size_bytes);
    noc_async_write_barrier();

    noc_semaphore_inc(rightB_semaphore_noc_addr, 1);

    // cleanup
    noc_semaphore_wait(A_semaphore_ptr, 1); // wait A = 1
    noc_semaphore_set(A_semaphore_ptr, 0);  // reset A for next round
  }

  if (have_left) {
    DPRINT_DATA1(DPRINT << "Waiting receive from left: " << B_semaphore_addr
                        << ENDL());
    noc_semaphore_wait(B_semaphore_ptr, 1); // wait B = 1
    noc_semaphore_set(B_semaphore_ptr, 0);  // reset for next round

    // cleanup for next round
    noc_semaphore_inc(leftA_semaphore_noc_addr, 1); // A = 1
    DPRINT_DATA1(DPRINT << "DONE receive from left: " << B_semaphore_addr
                        << ENDL());
  }
}

// Exchanges halo slices with left/right neighbors for transposed tile layout.
//
// Similar to halo_exchange, but performs per-row strided sends/receives so the
// transferred elements correspond to east/west neighbors in transposed storage.
// This is used when EW stencil components are staged after transpose.
void halo_exchange_transpose(const uint32_t have_left, uint32_t left_noc_x,
                             uint32_t left_noc_y, const uint32_t have_right,
                             uint32_t right_noc_x, const uint32_t right_noc_y,
                             uint32_t A_semaphore_addr,
                             uint32_t B_semaphore_addr, uint32_t size_bytes,
                             int halo_size, uint32_t cb_shifted_tiles_ptr2,
                             uint32_t cb_shifted_tiles_ptr3) {

  volatile tt_l1_ptr uint32_t *A_semaphore_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(A_semaphore_addr);
  volatile tt_l1_ptr uint32_t *B_semaphore_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(B_semaphore_addr);

  uint64_t leftA_semaphore_noc_addr;

  // send leftwards and receive from right
  uint64_t rightB_semaphore_noc_addr;
  if (have_right) {
    rightB_semaphore_noc_addr =
        get_noc_addr(right_noc_x, right_noc_y, B_semaphore_addr);
    noc_semaphore_inc(rightB_semaphore_noc_addr,
                      1); // ready to receive from right
  }
  if (have_left) {
    leftA_semaphore_noc_addr =
        get_noc_addr(left_noc_x, left_noc_y, A_semaphore_addr);

    noc_semaphore_wait(B_semaphore_ptr, 1); // wait B = 1
    noc_semaphore_set(B_semaphore_ptr, 0);  // reset for next round

    for (int row = 0; row < 4; row++) {
      uint32_t row_offset = row * halo_size * size_bytes;
      uint32_t left_dst_addr =
          cb_shifted_tiles_ptr3 + row_offset + (halo_size - 1) * size_bytes;
      uint32_t left_src_l1_addr =
          cb_shifted_tiles_ptr2 + row_offset + size_bytes;

      uint64_t left_dst_noc_addr =
          get_noc_addr(left_noc_x, left_noc_y, left_dst_addr);
      noc_async_write(left_src_l1_addr, left_dst_noc_addr, size_bytes);
    }
    noc_async_write_barrier();

    noc_semaphore_inc(leftA_semaphore_noc_addr, 1); // A = 1
  }

  if (have_right) {
    DPRINT_DATA1(DPRINT << "Waiting receive from right: " << A_semaphore_addr
                        << ENDL());
    noc_semaphore_wait(A_semaphore_ptr, 1); // wait for send to complete
    noc_semaphore_set(A_semaphore_ptr, 0);  // reset A

    rightB_semaphore_noc_addr =
        get_noc_addr(right_noc_x, right_noc_y, B_semaphore_addr);

    DPRINT_DATA1(DPRINT << "Sending right: waiting at A: " << A_semaphore_addr
                        << " and alerting at A: " << A_semaphore_addr
                        << ENDL());

    for (int row = 0; row < 4; row++) {
      uint32_t row_offset = row * halo_size * size_bytes;
      uint32_t right_dst_addr = cb_shifted_tiles_ptr2 + row_offset;
      uint32_t right_src_l1_addr =
          cb_shifted_tiles_ptr3 + row_offset + (halo_size - 2) * size_bytes;
      uint64_t right_dst_noc_addr =
          get_noc_addr(right_noc_x, right_noc_y, right_dst_addr);
      noc_async_write(right_src_l1_addr, right_dst_noc_addr, size_bytes);
    }

    noc_async_write_barrier();

    noc_semaphore_inc(rightB_semaphore_noc_addr, 1);

    // cleanup
    noc_semaphore_wait(A_semaphore_ptr, 1); // wait A = 1
    noc_semaphore_set(A_semaphore_ptr, 0);  // reset A for next round
  }

  if (have_left) {
    DPRINT_DATA1(DPRINT << "Waiting receive from left: " << B_semaphore_addr
                        << ENDL());
    noc_semaphore_wait(B_semaphore_ptr, 1); // wait B = 1
    noc_semaphore_set(B_semaphore_ptr, 0);  // reset for next round

    // cleanup for next round
    noc_semaphore_inc(leftA_semaphore_noc_addr, 1); // A = 1
    DPRINT_DATA1(DPRINT << "DONE receive from left: " << B_semaphore_addr
                        << ENDL());
  }
}
