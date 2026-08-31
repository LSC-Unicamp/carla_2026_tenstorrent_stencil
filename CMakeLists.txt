cmake_minimum_required(VERSION 3.22...3.30)

project(jacobi_deslocamento_sistolico LANGUAGES CXX)

add_executable(jacobi_deslocamento_sistolico)
target_sources(jacobi_deslocamento_sistolico PRIVATE host.cpp)
target_compile_definitions(jacobi_deslocamento_sistolico PRIVATE JACOBI_KERNEL_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

set_target_properties(jacobi_deslocamento_sistolico PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

if(NOT DEFINED ENV{TT_METAL_HOME})
    message(FATAL_ERROR "TT_METAL_HOME is not defined. Export it to your tt-metal checkout before configuring this project.")
endif()

if(NOT TARGET TT::Metalium AND NOT DEFINED TT-Metalium_DIR)
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{TT_METAL_HOME}/build/lib/cmake" "$ENV{TT_METAL_HOME}/build/share/cmake" "$ENV{TT_METAL_HOME}/build")
    set(_tt_metalium_dir "$ENV{TT_METAL_HOME}/build/lib/cmake/tt-metalium")
    if(EXISTS "${_tt_metalium_dir}/tt-metalium-config.cmake")
        set(TT-Metalium_DIR "${_tt_metalium_dir}")
    endif()
endif()

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(jacobi_deslocamento_sistolico PUBLIC TT::Metalium)
