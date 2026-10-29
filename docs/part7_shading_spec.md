\# Part 7: Shading and Material Specification



\## 1. Introduction



The objective of Part 7 is to define the shading and material requirements for the BVH ray-tracing project. The scene system supports different material types to represent objects with different light-interaction properties. These materials include diffuse, mirror, glass, and emissive materials.



The purpose of this specification is to describe the expected behavior of each material, the role of surface normals, and how light transport contributes to the final rendered image.



\## 2. Material Types



\### 2.1 Diffuse Material



A diffuse material represents a matte surface that scatters incoming light in multiple directions. Examples include walls, floors, and unpolished objects.



The appearance of a diffuse surface depends on its base color, surface orientation, and illumination. A surface facing a light source generally receives more light than a surface facing away from it.



The following properties are relevant to diffuse shading:



\- \*\*Base color:\*\* Determines the color of the surface.

\- \*\*Surface normal:\*\* Determines the orientation of the surface.

\- \*\*Lighting:\*\* Determines the amount of illumination received by the surface.



Diffuse materials are used for the floor, walls, and other non-reflective objects in the test scenes.



\### 2.2 Mirror Material



A mirror material represents a reflective surface. When a ray strikes a mirror, the expected behavior is to generate a reflected ray based on the incoming ray direction and the surface normal.



The reflection direction can be expressed as:



\\\[

R = D - 2(D \\cdot N)N

\\]



where:



\- \\(D\\) is the normalized incoming ray direction.

\- \\(N\\) is the normalized surface normal.

\- \\(R\\) is the reflected ray direction.



A perfect mirror reflects rays without roughness. If the renderer supports a fuzz or roughness parameter, the reflection may be perturbed to represent an imperfect reflective surface.



Mirror materials are useful for testing reflections and the interaction between reflective objects and their surroundings.



\### 2.3 Glass Material



A glass material represents a transparent surface through which light can travel. When a ray hits glass, its behavior depends on the angle of incidence and the refractive properties of the material.



The two important optical effects are:



\- \*\*Reflection:\*\* A portion of the light travels back into the surrounding medium.

\- \*\*Refraction:\*\* Light changes direction as it passes between materials with different refractive indices.



The refractive index determines how strongly the light bends. For example, an index of refraction of 1.5 is a commonly used approximation for glass.



Glass shading should account for whether a ray enters or exits the object. The renderer should use appropriately oriented surface normals and distinguish front-face and back-face intersections when calculating refraction.



Total internal reflection may occur under certain conditions. If supported by the implementation, the renderer should handle it by reflecting the ray instead of producing an invalid refracted direction.



Glass materials are used to test transparency, refraction, and reflection.



\### 2.4 Emissive Material



An emissive material represents a surface that emits light. Unlike ordinary diffuse and reflective materials, its defining property is that it contributes emitted radiance.



The emission color and intensity determine the light produced by the surface. Higher emission intensity can make a surface appear brighter.



Emissive geometry can represent light sources in the scene. The renderer may also use emissive triangles as area lights for direct-light sampling, depending on its implementation.



For the existing test scene, the lamp material is specified as:



`material lamp emissive 12 12 12`



This defines an emissive material with equal RGB intensity values.



\## 3. Surface Normals



A surface normal is a vector perpendicular to a surface. Normals are essential for calculating reflection, refraction, and the effect of lighting on a surface.



The project uses triangle meshes to represent geometry. Each triangle has a geometric normal derived from its vertices. Meshes may also contain vertex normals, which can be interpolated to produce smoother shading across a surface.



Two types of normals are relevant:



\- \*\*Geometric normals:\*\* Represent the orientation of a triangle.

\- \*\*Shading normals:\*\* Determine the orientation used for shading calculations.



For rays that hit either side of a triangle, the normal may need to be flipped so that it faces against the incoming ray, depending on the operation being performed. Glass calculations must preserve enough information to distinguish entering and exiting rays.



Correct normal handling helps avoid incorrect reflections, refractions, and lighting artifacts.



\## 4. Ray Bounces and Light Transport



Ray tracing simulates the interaction between rays and objects in a scene. When a ray intersects a surface, the material determines how the renderer handles the interaction.



Depending on the material, the renderer may evaluate surface lighting, generate a reflected ray, generate a refracted ray, or account for emitted light.



The maximum bounce count limits how many successive ray interactions are evaluated. A higher bounce count may capture more indirect light transport and additional reflections or refractions, but it can also increase rendering time.



The scene configuration supports the following setting:



`max\_bounces 5`



This specifies a maximum bounce limit of five for the scene. The exact counting convention and the behavior when the limit is reached depend on the renderer implementation.



\## 5. Scene Configuration



The scene description file defines the image dimensions, sampling settings, camera, environment, materials, and geometry.



For example, the existing sphere scene includes the following material definitions:



```text

material matte diffuse 0.8 0.3 0.25

material metal mirror 0.9 0.9 0.95 fuzz 0.05

material glass glass 1.5

material floor diffuse 0.6 0.6 0.6

material lamp emissive 12 12 12

```



These definitions demonstrate the configuration of diffuse, mirror, glass, and emissive materials. The exact meaning of individual parameters is governed by the project's scene format and material implementation.



The scene also specifies the camera, environment, image resolution, samples per pixel, and mesh geometry. Keeping these properties in scene files makes it possible to test different material combinations without modifying the renderer source code.



\## 6. Rendering and Color Handling



The project follows a linear-RGB workflow inside the renderer. Lighting calculations should be performed in linear color space, while tone mapping and gamma encoding, where applicable, belong to the output stage.



This separation is important because lighting calculations performed directly on gamma-encoded values can produce incorrect brightness and color results.



The final appearance of a material depends on its color, illumination, surface orientation, and the supported reflection or refraction behavior.



\## 7. Testing and Validation



The following tests are recommended for validating material and shading behavior.



| Test | Expected observation |

|---|---|

| Diffuse surface | Surface appearance responds to its color and illumination. |

| Mirror surface | Reflected rays reproduce the surrounding scene where visible. |

| Glass surface | Transparency and refraction are visible when supported by the renderer. |

| Emissive surface | The surface contributes emitted radiance and appears bright. |

| Normal handling | Surface orientation produces consistent shading and ray directions. |

| Multiple bounces | The configured bounce limit controls successive ray interactions. |

| Scene loading | Material definitions and mesh references are parsed correctly. |

| Complex geometry | Materials and normals remain usable on detailed triangle meshes. |



The existing sphere scene provides a useful starting point because it contains diffuse, mirror, glass, and emissive materials. Additional showcase scenes, including a Cornell box and a detailed model, can be used to evaluate these properties under more varied geometric and lighting conditions.



\## 8. Performance Considerations



The number of triangles in a scene affects the cost of ray-intersection queries. The project includes scenes with approximately 1,000, 50,000, and 500,000 triangles to examine behavior at different geometry sizes.



Performance testing should distinguish scene loading and BVH construction from rendering time. Increasing the sample count, resolution, or bounce limit may also increase rendering cost.



Material complexity can affect performance because reflection and refraction may require additional ray evaluations. Actual performance depends on the renderer implementation, acceleration structure, scene complexity, and sampling configuration.



\## 9. Conclusion



This specification establishes the intended shading behavior for diffuse, mirror, glass, and emissive materials in the BVH ray-tracing project. It also describes the importance of surface normals, ray bounces, scene configuration, linear color handling, testing, and performance evaluation.



The specification serves as a reference for implementing, testing, and documenting the shading behavior of the renderer. Any behavior not yet supported by the current implementation should be treated as a requirement for verification rather than an already completed feature.
