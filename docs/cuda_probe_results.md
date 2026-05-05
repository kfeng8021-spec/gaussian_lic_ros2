# CUDA Probe Results

Last verified: 2026-05-05 after rebuilding `gaussian_lic_mapping`.

| Probe | Result |
| --- | --- |
| `simple_knn_probe` | PASS, `max_abs_error=9.31323e-10` |
| `fused_ssim_probe` | PASS, `forward_value_error=6.57514e-07`, `gradient_error=3.49246e-09` |
| `rasterizer_probe` | PASS, rendered 24 buckets with finite gradients |
| `sparse_adam_probe` | PASS, parameter/state errors all `0`, elapsed `2.14198 ms` for 100k Gaussians x 4 parameters |
| `torch_cuda_backend_probe` | PASS, CUDA rasterize/optimize/densify/prune/reset path exercised |
| `torch_backend_probe` | PASS, libtorch `2.7.0`, CUDA available, tensor init/optimization/append/prune path exercised |

These probes verify the local CUDA operator layer is callable on the current
machine. The strict paper gate still depends on full-sequence final-map render
pairs and ROS1 baseline parity.
