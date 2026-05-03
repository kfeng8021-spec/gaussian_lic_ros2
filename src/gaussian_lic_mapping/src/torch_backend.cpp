// SPDX-License-Identifier: GPL-3.0-or-later

#include <gaussian_lic_mapping/torch_backend.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gaussian_lic_mapping
{
namespace
{

torch::Tensor eigen_matrix_to_torch_tensor(
  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matrix,
  torch::Device device)
{
  matrix.transposeInPlace();
  return torch::from_blob(
    matrix.data(),
    {matrix.rows(), matrix.cols()},
    torch::TensorOptions().dtype(torch::kFloat32)).clone().to(device);
}

torch::Tensor make_world_view_transform(
  const Eigen::Matrix3d & r_wc,
  const Eigen::Vector3d & t_wc,
  torch::Device device)
{
  Eigen::Matrix4f rt = Eigen::Matrix4f::Zero();
  const Eigen::Matrix3f r_cw = r_wc.transpose().cast<float>();
  const Eigen::Vector3f t_cw = (-r_wc.transpose() * t_wc).cast<float>();
  rt.topLeftCorner<3, 3>() = r_cw;
  rt.topRightCorner<3, 1>() = t_cw;
  rt(3, 3) = 1.0F;

  return eigen_matrix_to_torch_tensor(rt, device).transpose(0, 1);
}

torch::Tensor make_projection_matrix(
  const int width,
  const int height,
  const float fx,
  const float fy,
  const float cx,
  const float cy,
  torch::Device device)
{
  constexpr float z_near = 0.01F;
  constexpr float z_far = 100.0F;
  const float fov_x = 2.0F * std::atan(static_cast<float>(width) / (2.0F * fx));
  const float fov_y = 2.0F * std::atan(static_cast<float>(height) / (2.0F * fy));

  torch::Tensor projection = torch::zeros({4, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  projection.index_put_({0, 0}, 1.0F / std::tan(fov_x / 2.0F));
  projection.index_put_({1, 1}, 1.0F / std::tan(fov_y / 2.0F));
  projection.index_put_({0, 2}, (2.0F * cx - static_cast<float>(width)) / static_cast<float>(width));
  projection.index_put_({1, 2}, (2.0F * cy - static_cast<float>(height)) / static_cast<float>(height));
  projection.index_put_({3, 2}, 1.0F);
  projection.index_put_({2, 2}, z_far / (z_far - z_near));
  projection.index_put_({2, 3}, -(z_far * z_near) / (z_far - z_near));
  return projection.transpose(0, 1);
}

float rgb_to_sh(const float value)
{
  constexpr float c0 = 0.28209479177387814F;
  return (value - 0.5F) / c0;
}

torch::Tensor inverse_sigmoid(const torch::Tensor & x)
{
  return torch::log(x / (1.0F - x));
}

}  // namespace

torch::Tensor cv_mat_to_torch_tensor_float32(
  const cv::Mat & mat,
  torch::Device device,
  const bool use_pinned_memory)
{
  if (mat.empty()) {
    throw std::runtime_error("cannot convert empty cv::Mat to torch tensor");
  }
  if (mat.depth() != CV_32F) {
    throw std::runtime_error("cv_mat_to_torch_tensor_float32 expects CV_32F input");
  }

  torch::Tensor tensor;
  if (mat.channels() == 1) {
    tensor = torch::from_blob(
      const_cast<float *>(mat.ptr<float>()),
      {mat.rows, mat.cols},
      torch::TensorOptions().dtype(torch::kFloat32)).clone();
  } else if (mat.channels() == 3) {
    tensor = torch::from_blob(
      const_cast<float *>(mat.ptr<float>()),
      {mat.rows, mat.cols, mat.channels()},
      torch::TensorOptions().dtype(torch::kFloat32)).clone().permute({2, 0, 1});
  } else {
    throw std::runtime_error("unsupported cv::Mat channel count for torch conversion");
  }

  if (use_pinned_memory && device.is_cpu()) {
    tensor = tensor.pin_memory();
  }
  return tensor.to(device, /*non_blocking=*/use_pinned_memory).contiguous();
}

TorchCamera make_torch_camera(
  const CameraFrameRecord & frame,
  const double fx,
  const double fy,
  const double cx,
  const double cy,
  torch::Device image_device,
  torch::Device geometry_device)
{
  TorchCamera camera;
  camera.image_name = frame.image_name;
  camera.image_width = frame.width;
  camera.image_height = frame.height;
  camera.fx = static_cast<float>(fx);
  camera.fy = static_cast<float>(fy);
  camera.cx = static_cast<float>(cx);
  camera.cy = static_cast<float>(cy);
  camera.fov_x = 2.0F * std::atan(static_cast<float>(frame.width) / (2.0F * camera.fx));
  camera.fov_y = 2.0F * std::atan(static_cast<float>(frame.height) / (2.0F * camera.fy));

  camera.original_image = cv_mat_to_torch_tensor_float32(
    frame.image_rgb_float, image_device, frame.is_keyframe && image_device.is_cpu());
  camera.original_depth = cv_mat_to_torch_tensor_float32(
    frame.depth_m_float, image_device, frame.is_keyframe && image_device.is_cpu());

  camera.world_view_transform = make_world_view_transform(frame.r_wc, frame.t_wc, geometry_device);
  camera.projection_matrix = make_projection_matrix(
    frame.width, frame.height, camera.fx, camera.fy, camera.cx, camera.cy, geometry_device);
  camera.full_proj_transform = (
    camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0);
  camera.camera_center = camera.world_view_transform.inverse().index({3, torch::indexing::Slice(0, 3)});
  return camera;
}

TorchGaussianMap initialize_gaussian_map(
  const MapperDataset & dataset,
  const int sh_degree,
  const double scaling_scale,
  const double fx,
  const double fy,
  torch::Device device)
{
  const size_t num_points = dataset.pending_point_count();
  if (num_points == 0) {
    throw std::runtime_error("cannot initialize Gaussian map without pending points");
  }
  if (sh_degree < 0) {
    throw std::runtime_error("sh_degree must be non-negative");
  }
  if (scaling_scale <= 0.0) {
    throw std::runtime_error("scaling_scale must be positive");
  }

  const double focal = (fx + fy) * 0.5;
  if (focal <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error("average focal length must be positive");
  }

  const int64_t num = static_cast<int64_t>(num_points);
  const int64_t sh_coeff_count = static_cast<int64_t>(sh_degree + 1) *
    static_cast<int64_t>(sh_degree + 1);

  const auto & points = dataset.pending_points_world();
  const auto & colors = dataset.pending_colors_rgb();
  const auto & depths = dataset.pending_depths_m();
  if (colors.size() != num_points || depths.size() != num_points) {
    throw std::runtime_error("pending point/color/depth buffers have inconsistent sizes");
  }

  std::vector<float> xyz_values(num_points * 3U);
  std::vector<float> feature_values(num_points * 3U * static_cast<size_t>(sh_coeff_count), 0.0F);
  std::vector<float> scale_values(num_points);

  for (size_t i = 0; i < num_points; ++i) {
    xyz_values[i * 3U + 0U] = points[i].x();
    xyz_values[i * 3U + 1U] = points[i].y();
    xyz_values[i * 3U + 2U] = points[i].z();

    const size_t feature_offset = i * 3U * static_cast<size_t>(sh_coeff_count);
    feature_values[feature_offset + 0U * static_cast<size_t>(sh_coeff_count)] =
      rgb_to_sh(colors[i].x());
    feature_values[feature_offset + 1U * static_cast<size_t>(sh_coeff_count)] =
      rgb_to_sh(colors[i].y());
    feature_values[feature_offset + 2U * static_cast<size_t>(sh_coeff_count)] =
      rgb_to_sh(colors[i].z());

    const double clamped_depth = std::max(static_cast<double>(depths[i]), 1e-6);
    scale_values[i] = static_cast<float>(std::log(scaling_scale * clamped_depth / focal));
  }

  const auto cpu_options = torch::TensorOptions().dtype(torch::kFloat32);
  auto xyz = torch::from_blob(xyz_values.data(), {num, 3}, cpu_options).clone().to(device);
  auto features = torch::from_blob(
    feature_values.data(), {num, 3, sh_coeff_count}, cpu_options).clone().to(device);
  auto scaling = torch::from_blob(scale_values.data(), {num}, cpu_options)
    .clone()
    .to(device)
    .unsqueeze(1)
    .repeat({1, 3});

  auto rotation = torch::zeros({num, 4}, cpu_options.device(device));
  rotation.index_put_({torch::indexing::Slice(), 0}, 1.0F);
  auto opacity = inverse_sigmoid(
    0.1F * torch::ones({num, 1}, cpu_options.device(device)));

  TorchGaussianMap map;
  map.xyz = xyz.requires_grad_();
  map.features_dc = features.index({
      torch::indexing::Slice(),
      torch::indexing::Slice(),
      torch::indexing::Slice(0, 1)})
    .transpose(1, 2)
    .contiguous()
    .requires_grad_();
  map.features_rest = features.index({
      torch::indexing::Slice(),
      torch::indexing::Slice(),
      torch::indexing::Slice(1, features.size(2))})
    .transpose(1, 2)
    .contiguous()
    .requires_grad_();
  map.scaling = scaling.requires_grad_();
  map.rotation = rotation.requires_grad_();
  map.opacity = opacity.requires_grad_();
  map.sh_degree = sh_degree;
  map.foreground_count = num_points;
  map.skybox_count = 0;
  return map;
}

}  // namespace gaussian_lic_mapping
