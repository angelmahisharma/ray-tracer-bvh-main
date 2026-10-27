// scene_stats <file.scene> [--csv]
#include <cstring>
#include <iostream>
#include "rt/scene.h"

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: scene_stats <file.scene> [--csv]\n"; return 1; }
    rt::Scene scene;
    std::string err;
    if (!rt::load_scene(argv[1], scene, &err)) { std::cerr << "error: " << err << "\n"; return 1; }
    if (argc > 2 && !strcmp(argv[2], "--csv")) {
        std::cout << rt::scene_stats_csv_header() << "\n" << rt::scene_stats_csv_row(scene) << "\n";
    } else {
        rt::print_scene_stats(scene, std::cout);
    }
    return 0;
}
