#include "render.cuh"
#include "path_tracer.h"

namespace rt {

__global__ void render_kernel(SceneView sc, float4* out, int y0, int rows) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= sc.width || y >= y0 + rows || y >= sc.height) return;
    Color c = render_pixel(sc, x, y);
    out[(size_t)y * sc.width + x] = make_float4(c.x, c.y, c.z, 1.f);
}

void launch_render(const SceneView& scene, float4* out, int y0, int rows, cudaStream_t stream) {
    dim3 block(8, 8);
    dim3 grid((scene.width + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
    render_kernel<<<grid, block, 0, stream>>>(scene, out, y0, rows);
}

}  // namespace rt
