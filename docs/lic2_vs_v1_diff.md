# Gaussian-LIC2 vs Public V1 Surface Diff

As of 2026-05-04, `APRIL-ZJU/Gaussian-LIC` exposes one public branch/tag surface
in this workspace:

```text
external/Gaussian-LIC @ cd4c122dfad7e93255fe6862ac2c2b205e844786
```

The upstream README announces Gaussian-LIC2, but the repository does not expose
a separate `Gaussian-LIC2` branch or tag to diff against the earlier Gaussian-LIC
surface. The actionable migration baseline is therefore the checked public tree.

## Mapping Backend

Files audited:

```text
src/mapping.cpp
src/mapping.h
src/gaussian.cpp
src/gaussian.h
src/rasterizer/*
src/simple-knn/*
src/fused-ssim/*
src/depth_completer.*
```

Observable public surface:

- Gaussian tensors remain 3D: `xyz[N,3]`, `scaling[N,3]`, `rotation[N,4]`.
- No released mapper code defines a 2D Gaussian primitive, surface-normal tensor,
  or normal-consistency loss.
- Skybox seeding remains the sampled sphere implementation in
  `GaussianModel::initialize()`.
- The CUDA rasterizer, fused SSIM, simple-knn, SparseGaussianAdam, and TensorRT
  depth-completion wrapper are the released LIC2 mapper backend surface.

## ROS2 Port Status

Ported in `gaussian_lic_mapping`:

- simple-knn `distCUDA2`
- fused-SSIM forward/backward
- Gaussian rasterizer forward/backward
- visibility-masked SparseGaussianAdam
- CUDA rasterizer loss and preview path
- gradient-aware density control
- optional TensorRT/SPNet depth-completion wrapper

No additional LIC2-only 2D primitive patch is currently available in the public
upstream tree. If upstream publishes a later LIC2 tag with 2D/surface-aligned
Gaussian changes, that tag must be diffed against `cd4c122` and this document
updated before claiming P4 parity.
