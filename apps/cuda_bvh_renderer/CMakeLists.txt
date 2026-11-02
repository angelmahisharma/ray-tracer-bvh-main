# CUDA path tracer with a CPU-built, GPU-traversed BVH (Part 6 bonus: GPU BVH without RT cores).
# The BVH builder and a CPU test of the traversal + path tracer always build;
# the GPU renderer is skipped when no CUDA toolkit is available.
add_library(cuda_bvh STATIC bvh_build.cpp)
target_include_directories(cuda_bvh PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cuda_bvh PUBLIC rt::common)

if(RT_BUILD_TESTS)
  add_executable(test_cuda_bvh test_bvh.cpp)
  target_link_libraries(test_cuda_bvh PRIVATE cuda_bvh)
  add_test(NAME cuda_bvh COMMAND test_cuda_bvh ${PROJECT_SOURCE_DIR}/scenes)
endif()

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
  message(STATUS "cuda_bvh_renderer: CUDA compiler not found - GPU renderer skipped (BVH test still built)")
  return()
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(cuda_bvh_renderer main.cpp render.cu)
target_link_libraries(cuda_bvh_renderer PRIVATE cuda_bvh CUDA::cudart)
