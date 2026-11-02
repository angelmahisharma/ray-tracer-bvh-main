# OptiX renderer (Parts 5 + 6). Needs the CUDA toolkit and the OptiX SDK
# (headers only). Skipped with a message when either is missing, so the rest
# of the project still configures on machines without them.
find_package(CUDAToolkit QUIET)
if(NOT CUDAToolkit_FOUND)
  message(STATUS "optix_renderer: CUDA toolkit not found - skipped")
  return()
endif()

set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}" CACHE PATH "OptiX SDK root (contains include/optix.h)")
if(NOT OptiX_INSTALL_DIR)
  file(GLOB _optix_sdks "C:/ProgramData/NVIDIA Corporation/OptiX SDK *")
  if(_optix_sdks)
    list(SORT _optix_sdks)
    list(REVERSE _optix_sdks)
    list(GET _optix_sdks 0 _optix_guess)
    set(OptiX_INSTALL_DIR "${_optix_guess}" CACHE PATH "OptiX SDK root (contains include/optix.h)" FORCE)
  endif()
endif()
find_path(OptiX_INCLUDE_DIR optix.h HINTS "${OptiX_INSTALL_DIR}/include" NO_DEFAULT_PATH)
if(NOT OptiX_INCLUDE_DIR)
  message(STATUS "optix_renderer: OptiX SDK not found (set OptiX_INSTALL_DIR) - skipped")
  return()
endif()
message(STATUS "optix_renderer: OptiX headers in ${OptiX_INCLUDE_DIR}")

set(OPTIX_RT_ARCH "sm_75" CACHE STRING "nvcc -arch for the OptiX device programs (OptiX-IR is JIT-compiled for the actual GPU)")
option(OPTIX_MATCH_CPU "Consume random numbers exactly like cpu_renderer, for pixel-by-pixel comparison (slower)" OFF)

# ---- device programs: .cu -> OptiX-IR -> embedded byte array ----
set(_dev_src ${CMAKE_CURRENT_SOURCE_DIR}/programs.cu)
set(_gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(_optixir ${CMAKE_CURRENT_BINARY_DIR}/programs.optixir)
set(_optixir_h ${_gen_dir}/programs_optixir.h)
set(_dev_defs)
if(OPTIX_MATCH_CPU)
  list(APPEND _dev_defs -DRT_MATCH_CPU=1)
endif()
file(GLOB _rt_headers ${PROJECT_SOURCE_DIR}/include/rt/*.h)

add_custom_command(
  OUTPUT ${_optixir_h}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${_gen_dir}
  COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} -optix-ir -arch=${OPTIX_RT_ARCH} -std=c++17
          --use_fast_math -lineinfo ${_dev_defs}
          -I${OptiX_INCLUDE_DIR} -I${PROJECT_SOURCE_DIR}/include -I${CMAKE_CURRENT_SOURCE_DIR}
          ${_dev_src} -o ${_optixir}
  COMMAND ${CMAKE_COMMAND} -DINPUT=${_optixir} -DOUTPUT=${_optixir_h} -DSYMBOL=kProgramsOptixIR
          -P ${CMAKE_CURRENT_SOURCE_DIR}/embed_file.cmake
  DEPENDS ${_dev_src} ${CMAKE_CURRENT_SOURCE_DIR}/launch_params.h
          ${CMAKE_CURRENT_SOURCE_DIR}/embed_file.cmake ${_rt_headers}
  COMMENT "Compiling OptiX device programs to OptiX-IR"
  VERBATIM)

# ---- shared core (pipeline, GAS, denoiser, Renderer) ----
add_library(optix_core STATIC renderer.cpp device_scene.cpp optix_state.cpp gas.cpp denoiser.cpp ${_optixir_h})
target_include_directories(optix_core PUBLIC ${OptiX_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} PRIVATE ${_gen_dir})
target_link_libraries(optix_core PUBLIC rt::common CUDA::cudart ${CMAKE_DL_LIBS})
if(WIN32)
  target_compile_definitions(optix_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---- offline renderer (Part 8 CLI contract) ----
add_executable(optix_renderer main.cpp)
target_link_libraries(optix_renderer PRIVATE optix_core)

# ---- interactive viewer: OpenGL through the GLFW + glad sources that ship with the OptiX SDK ----
option(OPTIX_BUILD_VIEWER "Build optix_viewer (needs OpenGL)" ON)
set(_optix_support "${OptiX_INCLUDE_DIR}/../SDK/support")
find_package(OpenGL QUIET)
if(OPTIX_BUILD_VIEWER AND OpenGL_FOUND AND EXISTS "${_optix_support}/glfw-3.4/CMakeLists.txt"
   AND EXISTS "${_optix_support}/glad/glad.c")
  enable_language(C)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
  add_subdirectory("${_optix_support}/glfw-3.4" "${CMAKE_CURRENT_BINARY_DIR}/glfw" EXCLUDE_FROM_ALL)
  add_library(optix_glad STATIC "${_optix_support}/glad/glad.c")
  target_include_directories(optix_glad PUBLIC "${_optix_support}")
  target_link_libraries(optix_glad PUBLIC OpenGL::GL)

  add_executable(optix_viewer viewer.cpp)
  target_link_libraries(optix_viewer PRIVATE optix_core optix_glad glfw)
else()
  message(STATUS "optix_renderer: optix_viewer skipped (needs OpenGL and the OptiX SDK's SDK/support/glfw-3.4)")
endif()
