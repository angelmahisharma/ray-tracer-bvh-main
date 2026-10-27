cmake_minimum_required(VERSION 3.16)
project(bvh_raytracing LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---- shared library: link every renderer against rt::common ----
add_library(rt_common STATIC src/scene_loader.cpp src/scene_stats.cpp src/image.cpp)
add_library(rt::common ALIAS rt_common)
target_include_directories(rt_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(MSVC)
  target_compile_options(rt_common PRIVATE /W3)
else()
  target_compile_options(rt_common PRIVATE -Wall -Wextra)
endif()

# ---- tools ----
add_executable(scene_stats tools/scene_stats_main.cpp)
target_link_libraries(scene_stats PRIVATE rt::common)
add_executable(debug_view tools/debug_view.cpp)
target_link_libraries(debug_view PRIVATE rt::common)

# ---- tests ----
option(RT_BUILD_TESTS "Build unit tests" ON)
if(RT_BUILD_TESTS)
  enable_testing()
  add_executable(test_basic tests/test_basic.cpp)
  target_link_libraries(test_basic PRIVATE rt::common)
  add_test(NAME basic COMMAND test_basic ${CMAKE_CURRENT_SOURCE_DIR}/scenes)
endif()

# ---- renderer apps: each owner just drops a CMakeLists.txt into their folder ----
foreach(app cpu_renderer cuda_renderer optix_renderer benchmark)
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps/${app}/CMakeLists.txt)
    add_subdirectory(apps/${app})
  endif()
endforeach()
