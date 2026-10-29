# Renderer CLI Contract (Part 8)

Every renderer MUST accept:
    <renderer> <scene.scene> <output_prefix> [spp] [max_bounces]

And MUST:
- Write `<output_prefix>.pfm` (linear radiance — mandatory)
- Write `<output_prefix>.png` (display output — optional)
- Print render time to stdout in a parseable form:
      "Render time: <seconds> s"

Timing rules:
- Time ONLY the render kernel — not scene load, not image save.
- The runner does 1 warm-up + 5 timed runs per renderer per scene.
- Report median, min, and std.