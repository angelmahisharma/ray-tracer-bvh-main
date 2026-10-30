# Part 7 Scene Validation

## 1. Triangle-count scenes

| Scene | Expected triangles | Actual triangles | Status |
|---|---:|---:|---|
| part7_1k | 976 | Fill from scene_stats | Pending |
| part7_50k | 49900 | Fill from scene_stats | Pending |
| part7_500k | 498736 | Fill from scene_stats | Pending |

## 2. Cornell box showcase

- Scene file: scenes/part7_cornell/cornell.scene
- Meshes loaded: Fill in after testing
- Glass sphere: Present in scene
- Mirror sphere: Present in scene
- Emissive panel: Present in scene
- Scene loading: Fill in after testing

## 3. Detailed model showcase

- Model: Stanford Bunny
- Scene file: scenes/part7_model/model.scene
- Mesh loading: Fill in after testing
- Triangle count: Fill in from scene_stats
- Preview: Fill in after testing

## 4. Automated tests

- Command: ctest --test-dir build --output-on-failure
- Result: Fill in after running

## 5. Limitations and observations

Record any scene-loading errors, preview issues, unsupported material features,
or performance problems observed during testing.
