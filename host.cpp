#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fmt/ostream.h>
#include <fstream>
#include <string>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/tt_metal_profiler.hpp>
#include <utility>
#include <vector>

/** **How the Program Works**
 *
 * A matrix is received through the input.bin file in float format.
 *
 * This matrix is converted to bfloat16 and tiled.
 *
 * Next, we assign a Tensix core to be responsible for each tile. This Tensix
 * will read its tile from DRAM and send it to its neighbors.
 *
 * Then the iterations begin. After each iteration, the Tensix cores must
 * communicate to transfer the tiles among themselves.
 *
 * After the iterations, the Tensix cores send the tiles they were responsible
 * for back to the DRAM.
 */

using namespace tt::tt_metal;

#ifndef JACOBI_KERNEL_ROOT
#define JACOBI_KERNEL_ROOT ""
#endif

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

namespace {

std::filesystem::path example_root() {
  const std::filesystem::path standalone_root = JACOBI_KERNEL_ROOT;
  if (!standalone_root.empty()) {
    return standalone_root;
  }

  const std::filesystem::path prefix = OVERRIDE_KERNEL_PREFIX;
  if (!prefix.empty()) {
    return prefix /
           "Jacobi_Tenstorrent/jacobi-deslocamento/multi-core-sistolico";
  }
  return std::filesystem::path(__FILE__).parent_path();
}

std::string kernel_path(const std::filesystem::path &relative_path) {
  return (example_root() / relative_path).string();
}

} // namespace

// Returns a matrix from a file
std::vector<float>
read_matrix_from_file(const std::filesystem::path &filename) {
  std::ifstream file(filename, std::ios::binary);
  TT_FATAL(file.good(), "Não foi possível abrir {}", filename.string());
  file.seekg(0, std::ios::end);
  size_t size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<float> data(size / sizeof(float));
  file.read(reinterpret_cast<char *>(data.data()), size);
  return data;
}

// Writes a matrix to a file
void write_matrix_to_file(const std::filesystem::path &filename,
                          const std::vector<float> &data) {
  std::ofstream file(filename, std::ios::binary);
  TT_FATAL(file.good(), "Não foi possível abrir {} para escrita",
           filename.string());
  size_t size = data.size() * sizeof(float);
  file.write(reinterpret_cast<const char *>(data.data()), size);
}

int main(int argc, char **argv) {
  uint32_t num_iterations;
  uint32_t input_n_rows;
  uint32_t input_n_cols;
  uint32_t input_n_layers;
  if (argc > 4) {
    num_iterations = static_cast<uint32_t>(std::stoul(argv[1]));
    input_n_rows = static_cast<uint32_t>(std::stoul(argv[2]));
    input_n_cols = static_cast<uint32_t>(std::stoul(argv[3]));
    input_n_layers = static_cast<uint32_t>(std::stoul(argv[4]));
  } else {
    fmt::println("Usage: {} <num_iterations> <num rows> <num cols> <num "
                 "layers> [engine] [input path] [output path]",
                 argv[0]);
    fmt::println("\nDefaults");
    fmt::println("input path = ./input.bin");
    fmt::println("output path = ./output.bin");
    // SFPU abandoned
    fmt::println("engine = FPU [FPU / SFPU]");
    return 1;
  }
  const std::string engine = (argc > 5) ? argv[5] : "FPU";
  const std::filesystem::path input_path =
      (argc > 6) ? argv[6] : std::filesystem::path("input.bin");
  const std::filesystem::path output_path =
      (argc > 7) ? argv[7] : std::filesystem::path("output.bin");
  // bool pass = true;
  try {
    // INFO: Buffers

    TT_FATAL(num_iterations > 0, "Num iterations must be > 0. Currently: {}",
             num_iterations);

    fmt::print("Iterações: {}\n", num_iterations);
    fmt::print("Lendo {}\n", input_path.string());
    std::vector<float> in = read_matrix_from_file(input_path);

    TT_FATAL(in.size() == input_n_rows * input_n_cols * input_n_layers,
             "The input matrix size ({}) does not match the parameters passed "
             "height({}) x width({}) x layers({}) = {})",
             in.size(), input_n_rows, input_n_cols, input_n_layers,
             input_n_rows * input_n_cols * input_n_layers);

    TT_FATAL(input_n_rows % tt::constants::TILE_HEIGHT == 0,
             "Height {} is not a multiple of {} (tile height)", input_n_rows,
             tt::constants::TILE_HEIGHT);

    TT_FATAL(input_n_cols % tt::constants::TILE_WIDTH == 0,
             "Width {} is not a multiple of {} (tile width)", input_n_cols,
             tt::constants::TILE_WIDTH);

    constexpr uint32_t elements_per_tile =
        tt::constants::TILE_WIDTH * tt::constants::TILE_HEIGHT;

    // Now we create shift matrices L and U
    //
    // When a matrix is multiplied by them, their elements are shifted
    //
    // Ex: L✕A = A_{shifted downwards}

    fmt::println("Creating auxiliary matrices L, U, LL, and UU");

    std::vector<float> L(elements_per_tile, 0);
    std::vector<float> U(elements_per_tile, 0);
    std::vector<float> LL(elements_per_tile, 0);
    std::vector<float> UU(elements_per_tile, 0);

    // L needs to be filled with ones in the lower diagonal and U needs
    // to be filled with ones in the upper diagonal

    for (int idx = 0; idx < tt::constants::TILE_WIDTH - 1; idx++) {
      {
        int i = idx + 1;
        int j = idx;
        L[i * tt::constants::TILE_WIDTH + j] = 1.0f;
      }
      {
        int i = idx;
        int j = idx + 1;
        U[i * tt::constants::TILE_WIDTH + j] = 1.0f;
      }
    }

    // LL and UU shift the matrix all the way, leaving only one column or
    // one line.
    //
    // Therefore we only need a single one in the corner

    // LL: A single 1 at i = 31 e j=0
    LL[31 * tt::constants::TILE_WIDTH + 0] = 1.0f;

    // UU: A single 1 at i = 0 e j=31
    UU[31] = 1.0f;

    // INFO: Core Range

    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device =
        distributed::MeshDevice::create_unit_mesh(device_id);
    auto compute_grid = mesh_device->compute_with_storage_grid_size();
    fmt::println("Device grid: {}x{} = {} cores", compute_grid.x,
                 compute_grid.y, compute_grid.x * compute_grid.y);

    constexpr tt::tt_metal::CoreCoord core_begin = {0, 0};
    const auto tile_cols = input_n_cols / tt::constants::TILE_WIDTH;
    const auto tile_rows = input_n_rows / tt::constants::TILE_HEIGHT;
    // const auto tiles_per_layer = tile_cols * tile_rows;
    const tt::tt_metal::CoreCoord core_end = {tile_cols - 1, tile_rows - 1};

    tt::tt_metal::CoreRange all_cores =
        tt::tt_metal::CoreRange(core_begin, core_end);

    distributed::MeshWorkload workload;
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const uint32_t num_tiles = in.size() / elements_per_tile;
    constexpr uint32_t tile_size_bytes = sizeof(float) * elements_per_tile;
    const uint32_t dram_input_buffer_size = tile_size_bytes * num_tiles;
    constexpr uint32_t dram_LU_buffer_size = tile_size_bytes;
    fmt::print(
        "num_tiles {}, width: {}, height: {}, layers: {} num_iterations: {}\n",
        num_tiles, input_n_cols, input_n_rows, input_n_layers, num_iterations);

    // We'll need a buffer in DRAM for each:
    // - bufferA
    // - bufferB
    // - Shift Matrix L
    // - Shift Matrix U
    // - Shift Matrix LL
    // - Shift Matrix UU

    fmt::print("criando os buffers na DRAM\n");
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = BufferType::DRAM,
    };

    distributed::ReplicatedBufferConfig dram_input_buffer_config{
        .size = dram_input_buffer_size,
    };

    distributed::ReplicatedBufferConfig dram_LU_buffer_config{
        .size = dram_LU_buffer_size,
    };

    auto dram_buffer = distributed::MeshBuffer::create(
        dram_input_buffer_config, dram_config, mesh_device.get());

    auto L_dram_buffer = distributed::MeshBuffer::create(
        dram_LU_buffer_config, dram_config, mesh_device.get());
    auto U_dram_buffer = distributed::MeshBuffer::create(
        dram_LU_buffer_config, dram_config, mesh_device.get());
    auto LL_dram_buffer = distributed::MeshBuffer::create(
        dram_LU_buffer_config, dram_config, mesh_device.get());
    auto UU_dram_buffer = distributed::MeshBuffer::create(
        dram_LU_buffer_config, dram_config, mesh_device.get());

    Program program = CreateProgram();

    // INFO: Circular Buffers

    fmt::println("Creating Circular Buffers");
    tt::CBIndex cb_in1 = tt::CBIndex::c_0;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(tile_size_bytes * input_n_layers,
                             {{cb_in1, tt::DataFormat::Float32}})
            .set_page_size(cb_in1, tile_size_bytes));

    tt::CBIndex cb_in2 = tt::CBIndex::c_1;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(tile_size_bytes * input_n_layers,
                             {{cb_in2, tt::DataFormat::Float32}})
            .set_page_size(cb_in2, tile_size_bytes));

    tt::CBIndex cb_top = tt::CBIndex::c_2;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(tile_size_bytes * input_n_layers,
                             {{cb_top, tt::DataFormat::Float32}})
            .set_page_size(cb_top, tile_size_bytes));

    tt::CBIndex cb_bottom = tt::CBIndex::c_3;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(tile_size_bytes * input_n_layers,
                             {{cb_bottom, tt::DataFormat::Float32}})
            .set_page_size(cb_bottom, tile_size_bytes));

    tt::CBIndex cb_left = tt::CBIndex::c_4;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(tile_size_bytes * input_n_layers,
                             {{cb_left, tt::DataFormat::Float32}})
            .set_page_size(cb_left, tile_size_bytes));

    tt::CBIndex cb_right = tt::CBIndex::c_5;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(tile_size_bytes * input_n_layers,
                             {{cb_right, tt::DataFormat::Float32}})
            .set_page_size(cb_right, tile_size_bytes));

    tt::CBIndex cb_out = tt::CBIndex::c_16;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(2 * tile_size_bytes * input_n_layers,
                             {{cb_out, tt::DataFormat::Float32}})
            .set_page_size(cb_out, tile_size_bytes));

    // First L and then U
    tt::CBIndex cb_LU = tt::CBIndex::c_29;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(4 * tile_size_bytes,
                             {{cb_LU, tt::DataFormat::Float32}})
            .set_page_size(cb_LU, tile_size_bytes));

    // First LL and then UU
    tt::CBIndex cb_LLUU = tt::CBIndex::c_30;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(4 * tile_size_bytes,
                             {{cb_LLUU, tt::DataFormat::Float32}})
            .set_page_size(cb_LLUU, tile_size_bytes));

    // CB que é preenchido com zeros
    tt::CBIndex cb_zero = tt::CBIndex::c_31;
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(tile_size_bytes,
                             {{cb_zero, tt::DataFormat::Float32}})
            .set_page_size(cb_zero, tile_size_bytes));

    // INFO: Semaphores
    fmt::println("Creating Semaphores");

    //
    // **semaphore_reader**
    //
    // This semaphore tells neighbots that the Reader is ready to receive
    const uint32_t semaphore_reader = CreateSemaphore(program, all_cores, 0);

    //
    // **semaphore_writer**
    //
    // This semaphore tells neighbors that the Writer has sent them his tiles
    const uint32_t semaphore_writer = CreateSemaphore(program, all_cores, 0);

    // INFO: Programando os Cores
    fmt::print("Programming {} cores, {}\n", all_cores.size(), all_cores.str());

    std::vector<uint32_t> reader_compile_time_args = {
        cb_in1, cb_out, cb_LU, cb_LLUU, cb_top, cb_bottom, cb_left, cb_right,
    };
    TensorAccessorArgs(*dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*L_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*U_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*LL_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*UU_dram_buffer).append_to(reader_compile_time_args);
    auto reader = CreateKernel(
        program, kernel_path("kernels/worker/dataflow/read.cpp"), all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args = {
        cb_out, cb_top, cb_bottom, cb_left, cb_right,
    };
    TensorAccessorArgs(*dram_buffer).append_to(writer_compile_time_args);
    auto writer = CreateKernel(
        program, kernel_path("kernels/worker/dataflow/write.cpp"), all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default,
                           .compile_args = writer_compile_time_args});

    std::vector<uint32_t> compute_compile_time_args = {
        cb_in1, cb_in2,    cb_out,  cb_LU,    cb_LLUU,
        cb_top, cb_bottom, cb_left, cb_right, cb_zero,
    };

    std::string compute_kernel_path;

    if (engine == "FPU") {
      fmt::println("Utilizando a FPU");
      compute_kernel_path =
          kernel_path("kernels/worker/compute/compute_FPU.cpp");
    } else if (engine == "SFPU") {
      fmt::println("Utilizando a SFPU");
      compute_kernel_path =
          kernel_path("kernels/worker/compute/compute_SFPU.cpp");
    } else {
      TT_FATAL(false, "Unknown Engine {}. Use 'FPU' or 'SFPU'", engine);
    }

    auto compute = CreateKernel(program, compute_kernel_path, all_cores,
                                ComputeConfig{
                                    .math_fidelity = MathFidelity::HiFi4,
                                    .fp32_dest_acc_en = true,
                                    .math_approx_mode = false,
                                    .compile_args = compute_compile_time_args,
                                });

    uint32_t in_addr = dram_buffer->address();
    uint32_t L_addr = L_dram_buffer->address();
    uint32_t U_addr = U_dram_buffer->address();
    uint32_t LL_addr = LL_dram_buffer->address();
    uint32_t UU_addr = UU_dram_buffer->address();

    for (const auto &core : corerange_to_cores(all_cores)) {
      const auto physical_core =
          mesh_device->worker_core_from_logical_core(core);
      const uint32_t logical_x = core.x - all_cores.start_coord.x;
      const uint32_t logical_y = core.y - all_cores.start_coord.y;
      const uint32_t tile_offset = logical_y * tile_cols + logical_x;

      fmt::println("Creating RuntimeArgs for {}", core.str(), tile_offset);
      fmt::println("core {} logical=({}, {}) physical=({}, {}) tile_offset={} "
                   "tile_cols={} tile_rows={} layers={}",
                   core.str(), logical_x, logical_y,
                   static_cast<uint32_t>(physical_core.x),
                   static_cast<uint32_t>(physical_core.y), tile_offset,
                   tile_cols, tile_rows, input_n_layers);

      SetRuntimeArgs(program, reader, core,
                     {
                         in_addr,
                         tile_offset,
                         /* num_tiles =*/(uint32_t)1,
                         num_iterations,
                         L_addr,
                         U_addr,
                         LL_addr,
                         UU_addr,
                         semaphore_reader,
                         semaphore_writer,
                         logical_x,
                         logical_y,
                         tile_cols,
                         tile_rows,
                         input_n_layers,
                         static_cast<uint32_t>(physical_core.x),
                         static_cast<uint32_t>(physical_core.y),
                     });

      SetRuntimeArgs(program, writer, core,
                     {
                         in_addr,
                         tile_offset,
                         /*num_tiles =*/1,
                         num_iterations,
                         semaphore_reader,
                         semaphore_writer,
                         logical_x,
                         logical_y,
                         tile_cols,
                         tile_rows,
                         input_n_layers,
                         static_cast<uint32_t>(physical_core.x),
                         static_cast<uint32_t>(physical_core.y),
                     });

      SetRuntimeArgs(program, compute, core,
                     {
                         /* num_tiles= */ 1,
                         num_iterations,
                         logical_x,
                         logical_y,
                         tile_cols,
                         tile_rows,
                         input_n_layers,
                     });
    }

    // E escrevemos todos eles na DRAM do device
    fmt::println("Writing Buffers to DRAM");
    distributed::MeshCommandQueue &cq = mesh_device->mesh_command_queue();

    distributed::EnqueueWriteMeshBuffer(
        cq, dram_buffer, tilize_nfaces(in, input_n_rows, input_n_cols));
    distributed::EnqueueWriteMeshBuffer(
        cq, L_dram_buffer,
        tilize_nfaces(L, tt::constants::TILE_HEIGHT,
                      tt::constants::TILE_WIDTH));
    distributed::EnqueueWriteMeshBuffer(
        cq, U_dram_buffer,
        tilize_nfaces(U, tt::constants::TILE_HEIGHT,
                      tt::constants::TILE_WIDTH));
    distributed::EnqueueWriteMeshBuffer(
        cq, LL_dram_buffer,
        tilize_nfaces(LL, tt::constants::TILE_HEIGHT,
                      tt::constants::TILE_WIDTH));
    distributed::EnqueueWriteMeshBuffer(
        cq, UU_dram_buffer,
        tilize_nfaces(UU, tt::constants::TILE_HEIGHT,
                      tt::constants::TILE_WIDTH));

    fmt::println("Programming Device");

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);

    distributed::Finish(cq);
    std::vector<float> result_vec_tilized;

    fmt::print("Lendo resultado do buffer\n");
    distributed::EnqueueReadMeshBuffer(cq, result_vec_tilized, dram_buffer,
                                       true);

    std::vector<float> result_vec =
        untilize_nfaces(result_vec_tilized, input_n_rows, input_n_cols);
    ReadMeshDeviceProfilerResults(*mesh_device);

    mesh_device->close();

    fmt::print("Writing result to {}\n", output_path.string());
    write_matrix_to_file(output_path, result_vec);

  } catch (const std::exception &e) {
    fmt::print(stderr, "Test failed with exception! what: {}\n", e.what());
    throw;
  }
}
