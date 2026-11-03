#pragma once
#include <vector>
#include "launch_params.h"
#include "rt/scene.h"

struct DeviceSceneData {
    std::vector<TriShade>    tris;
    std::vector<DevMaterial> materials;
    std::vector<DevLight>    lights;
    std::vector<float>       light_cdf;
    std::vector<int32_t>     tri_to_light;
    std::vector<uint8_t>     tri_sbt_offset;   // material type per triangle (GAS build input)
};

DeviceSceneData pack_scene(const rt::Scene& scene);
