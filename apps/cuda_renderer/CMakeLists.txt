enable_language(CUDA)

add_executable(cuda_renderer main.cu)
target_link_libraries(cuda_renderer PRIVATE rt::common)

# Put YOUR GPU's compute capability here without the dot (7.5 -> 75, 8.6 -> 86).
set_target_properties(cuda_renderer PROPERTIES
    CUDA_ARCHITECTURES 75
    CUDA_STANDARD 17)