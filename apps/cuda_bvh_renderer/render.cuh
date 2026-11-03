#pragma once
#include <cuda_runtime.h>
#include "scene_view.h"

namespace rt {
// Renders rows [y0, y0 + rows) into out (width*height float4, row 0 = top).
// `scene` must hold DEVICE pointers. Asynchronous on `stream`.
void launch_render(const SceneView& scene, float4* out, int y0, int rows, cudaStream_t stream);
}
