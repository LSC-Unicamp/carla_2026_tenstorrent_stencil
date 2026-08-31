// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <random>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>

#include <tt-metalium/bfloat16.hpp>

#include "unit-tests.h"

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

/*
 * Host-side driver for the standalone 7-point stencil SpMV example.
 *
 * Workflow:
 * 1. Create one input vector tile-stack distributed across a 2D core grid.
 * 2. Stage the input tiles from DRAM into L1.
 * 3. Run the stencil program, which performs halo exchange and computes Ax.
 * 4. Write the result tiles back to DRAM and compare against the CPU golden.
 * */

/***********************************/

template <uint32_t tile_size>
// Creates one circular buffer for a program over the provided core set.
void create_cb(
    Program &program,
    const std::variant<CoreCoord, CoreRange, CoreRangeSet> &core_spec,
    const uint32_t index, uint32_t ntiles) {
  const uint32_t cb_x_index = index;
  CircularBufferConfig cb_x_config =
      CircularBufferConfig((ntiles)*tile_size, // arg1: total CB size
                                               // (for 2 tiles)
                           {{cb_x_index, tt::DataFormat::Float16_b}}
                           // arg2: map index to data format
                           )
          .set_page_size(cb_x_index,
                         tile_size); // chunk size in CB (set to tile size)
  tt_metal::CreateCircularBuffer(program, core_spec, cb_x_config);
}

bool driver(uint32_t num_input_tiles_per_core, size_t ncoresx, size_t ncoresy) {
  // ZoneScopedN("MainProgram");
  /*
   * Setup program
   */

  CoreCoord start_core = {0, 0};
  CoreCoord end_core = {ncoresx - 1, ncoresy - 1};

  constexpr int device_id = 0;
  std::shared_ptr<distributed::MeshDevice> mesh_device =
      distributed::MeshDevice::create_unit_mesh(device_id);

  Program read_from_dram_program = CreateProgram();
  Program spmv_program = CreateProgram();
  Program write_to_dram_program = CreateProgram();

  distributed::MeshWorkload read_workload;
  distributed::MeshWorkload spmv_workload;
  distributed::MeshWorkload write_workload;
  auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

  /*
   * Setup grid details
   */
  uint32_t start_core_x = start_core.x;
  uint32_t start_core_y = start_core.y;
  uint32_t num_cores_x = end_core.x - start_core.x + 1;
  uint32_t num_cores_y = end_core.y - start_core.y + 1;

  uint32_t end_core_x = end_core.x;
  uint32_t end_core_y = end_core.y;

  const CoreRange all_cores(
      {(std::size_t)start_core_x, (std::size_t)start_core_y},
      {(std::size_t)end_core_x, (std::size_t)end_core_y});

  uint32_t num_cores = all_cores.size();

  /*
   * Buffer setup
   */
  constexpr uint32_t single_tile_size =
      2 * 1024; // 32x32 tile, each FP16 is 2 bytes
  uint32_t num_tiles = num_input_tiles_per_core * num_cores;

  distributed::DeviceLocalBufferConfig dram_config{
      .page_size = single_tile_size,
      .buffer_type = BufferType::DRAM,
  };

  distributed::ReplicatedBufferConfig dram_buffer_config{
      .size = num_tiles * single_tile_size,
  };

  auto src_dram_buffer = distributed::MeshBuffer::create(
      dram_buffer_config, dram_config, mesh_device.get());
  auto dst_dram_buffer = distributed::MeshBuffer::create(
      dram_buffer_config, dram_config, mesh_device.get());

  /*
   * Create source data and write to DRAM.
   */
  std::vector<bfloat16> src0_vec(num_tiles * 1024);

  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> distribution(1.0f, 2.0f);

  for (auto &val : src0_vec) {
    val = bfloat16(distribution(rng));
  }

  /*
   * Create semaphores for spmv
   */
  auto east_semaphore_id =
      tt_metal::CreateSemaphore(spmv_program, all_cores, 0);
  auto west_semaphore_id =
      tt_metal::CreateSemaphore(spmv_program, all_cores, 0);
  auto north_semaphore_id =
      tt_metal::CreateSemaphore(spmv_program, all_cores, 0);
  auto south_semaphore_id =
      tt_metal::CreateSemaphore(spmv_program, all_cores, 0);

  /*
   * Create CBs
   */

  // reader
  create_cb<single_tile_size>(read_from_dram_program, all_cores,
                              tt::CBIndex::c_0, num_input_tiles_per_core);

  // spmv
  create_cb<single_tile_size>(spmv_program, all_cores, tt::CBIndex::c_0,
                              num_input_tiles_per_core);
  create_cb<single_tile_size>(spmv_program, all_cores, tt::CBIndex::c_1, 2);
  create_cb<single_tile_size>(spmv_program, all_cores, tt::CBIndex::c_2,
                              1); // scale by 6
  create_cb<single_tile_size>(spmv_program, all_cores, tt::CBIndex::c_3,
                              4); // shifted tiles
  create_cb<single_tile_size>(spmv_program, all_cores, tt::CBIndex::c_4,
                              0); // from reader
  create_cb<single_tile_size>(spmv_program, all_cores, tt::CBIndex::c_5,
                              2); // from reader
  create_cb<single_tile_size>(spmv_program, all_cores, tt::CBIndex::c_16,
                              num_input_tiles_per_core); // output

  // writer
  // dummy cb to get c_16 in the right place
  create_cb<single_tile_size>(write_to_dram_program, all_cores,
                              tt::CBIndex::c_0,
                              num_input_tiles_per_core + 2 + 1 + 4 + 2);
  create_cb<single_tile_size>(write_to_dram_program, all_cores,
                              tt::CBIndex::c_16,
                              num_input_tiles_per_core); // output

  /*
   * Create Kernels
   */

  // initial reader
  KernelHandle read_to_sram_kernel_id = CreateKernel(
      read_from_dram_program,
      OVERRIDE_KERNEL_PREFIX
      "dataflow/read-to-sram.cpp",
      all_cores,
      DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                         .noc = NOC::RISCV_1_default,
                         .compile_args = {}});

  // spmv kernels
  KernelHandle spmv_reader_kernel_id = CreateKernel(
      spmv_program,
      OVERRIDE_KERNEL_PREFIX
      "dataflow/spmv-reader.cpp",
      all_cores,
      DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                         .noc = NOC::RISCV_1_default,
                         .compile_args = {}});

  KernelHandle spmv_compute_kernel_id =
      CreateKernel(spmv_program,
                   OVERRIDE_KERNEL_PREFIX
                   "compute/spmv.cpp",
                   all_cores,
                   ComputeConfig{.math_fidelity = MathFidelity::HiFi4});
  SetRuntimeArgs(spmv_program, spmv_compute_kernel_id, all_cores,
                 {num_input_tiles_per_core, 0, 1, 2, 3, 4, 5, 16});

  // final writer
  KernelHandle write_from_sram_kernel_id = CreateKernel(
      write_to_dram_program,
      OVERRIDE_KERNEL_PREFIX
      "dataflow/write-from-sram.cpp",
      all_cores,
      DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                         .noc = NOC::RISCV_0_default,
                         .compile_args = {}});

  /*
   * Set Kernel arguments
   */
  for (uint32_t core_x = start_core_x; core_x <= end_core_x; core_x++) {
    for (uint32_t core_y = start_core_y; core_y <= end_core_y; core_y++) {
      CoreCoord core = {core_x, core_y};

      uint32_t my_start_input_tile =
          ((core_x - start_core_x) * num_cores_y + (core_y - start_core_y)) *
          num_input_tiles_per_core;

      uint32_t my_directions = 0;
      uint32_t north = 0x1;
      uint32_t south = 0x2;
      uint32_t west = 0x4;
      uint32_t east = 0x8;

      uint32_t west_noc_x = 0;
      uint32_t west_noc_y = 0;
      uint32_t east_noc_x = 0;
      uint32_t east_noc_y = 0;

      uint32_t north_noc_x = 0;
      uint32_t north_noc_y = 0;
      uint32_t south_noc_x = 0;
      uint32_t south_noc_y = 0;

      if (core_y != 0) {
        my_directions |= west;
        CoreCoord west_core = CoreCoord({core_x, core_y - 1});
        auto west_core_physical =
            mesh_device->worker_core_from_logical_core(west_core);

        west_noc_x = (std::uint32_t)west_core_physical.x;
        west_noc_y = (std::uint32_t)west_core_physical.y;
      }
      if (core_y != num_cores_y - 1) {
        my_directions |= east;
        CoreCoord east_core = CoreCoord({core_x, core_y + 1});
        auto east_core_physical =
            mesh_device->worker_core_from_logical_core(east_core);

        east_noc_x = (std::uint32_t)east_core_physical.x;
        east_noc_y = (std::uint32_t)east_core_physical.y;
      }
      if (core_x != 0) {
        my_directions |= north;
        CoreCoord north_core = CoreCoord({core_x - 1, core_y});
        auto north_core_physical =
            mesh_device->worker_core_from_logical_core(north_core);

        north_noc_x = (std::uint32_t)north_core_physical.x;
        north_noc_y = (std::uint32_t)north_core_physical.y;
      }
      if (core_x != num_cores_x - 1) {
        my_directions |= south;
        CoreCoord south_core = CoreCoord({core_x + 1, core_y});
        auto south_core_physical =
            mesh_device->worker_core_from_logical_core(south_core);

        south_noc_x = (std::uint32_t)south_core_physical.x;
        south_noc_y = (std::uint32_t)south_core_physical.y;
      }

      SetRuntimeArgs(read_from_dram_program, read_to_sram_kernel_id, core,
                     {num_input_tiles_per_core, // 1 tile per core
                      src_dram_buffer->address(), my_start_input_tile});

      SetRuntimeArgs(spmv_program, spmv_reader_kernel_id, core,
                     {num_input_tiles_per_core,
                      my_directions,
                      north_noc_x,
                      north_noc_y,
                      south_noc_x,
                      south_noc_y,
                      west_noc_x,
                      west_noc_y,
                      east_noc_x,
                      east_noc_y,
                      north_semaphore_id,
                      south_semaphore_id,
                      east_semaphore_id,
                      west_semaphore_id,
                      0,
                      1,
                      2,
                      3,
                      4,
                      16});

      SetRuntimeArgs(write_to_dram_program, write_from_sram_kernel_id, core,
                     {num_input_tiles_per_core, // 1 tile per core
                      dst_dram_buffer->address(), my_start_input_tile, 16});
    }
  }

  /*
   * Enqueue programs
   */
  distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();
  distributed::EnqueueWriteMeshBuffer(cq, src_dram_buffer, src0_vec);

  read_workload.add_program(device_range, std::move(read_from_dram_program));
  distributed::EnqueueMeshWorkload(cq, read_workload, false);

  spmv_workload.add_program(device_range, std::move(spmv_program));
  distributed::EnqueueMeshWorkload(cq, spmv_workload, false);

  write_workload.add_program(device_range, std::move(write_to_dram_program));
  distributed::EnqueueMeshWorkload(cq, write_workload, false);

  distributed::Finish(cq);

  /*
   * Read in result into a host vector.
   */
  std::vector<bfloat16> x_result_vec;
  distributed::EnqueueReadMeshBuffer(cq, x_result_vec, dst_dram_buffer, true);

  /*
   * Process results
   */

  bool pass = true;
  // std::vector<float> golden_result_vec;
  // host_spmv(ncoresx, ncoresy, num_input_tiles_per_core, src0_vec,
  //           golden_result_vec);

  // pass &= test_results(golden_result_vec, x_result_vec);
  pass &= mesh_device->close();

  return pass;
}

int main(int argc, char *argv[]) {
  if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
    TT_THROW("Test not supported w/ slow dispatch, exiting");
  }

  if (argc < 5) {
    printf("Usage: \n - #cores in x \n - #cores in y \n - #tiles per core "
           "\n - "
           "#iteration \n");
    return 1;
  }

  bool pass = true;

  try {
    // parse command line arguments
    uint32_t num_input_tiles_per_core = 1;
    [[maybe_unused]] uint32_t num_iters = 1;

    size_t ncoresx = std::atoi(argv[1]);
    size_t ncoresy = std::atoi(argv[2]);

    num_input_tiles_per_core = std::atoi(argv[3]);

    num_iters = std::atoi(argv[4]);

    // launch driver
    pass &= driver(num_input_tiles_per_core, ncoresx, ncoresy);

  } catch (const std::exception &e) {
    return 1;
  }

  if (pass) {
    std::cout << "Program completed." << std::endl;
  } else {
    TT_THROW("Program failed.");
  }

  return 0;
}
