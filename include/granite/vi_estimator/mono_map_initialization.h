/**
MIT License

This file is part of the Granite project.
https://github.com/DLR-RM/granite

Copyright (c) Martin Wudenka, Deutsches Zentrum für Luft- und Raumfahrt

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#pragma once

#include <granite/utils/assert.h>
#include <granite/vi_estimator/ba_base.h>
#include <granite/utils/eigen_utils.hpp>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <sophus/se3.hpp>

#include <optional>

namespace granite {

// compute homography through "Direct Linear Transformation" algorithm
// see "Multiple View Geometry in Computer Vision" 2nd Edition - Hartley,
// Zisserman, chapter 4.1
// use bearing vectors instead of 2D pixel coordinates and pinhole camera
std::optional<Eigen::Matrix3d> compute_homography(
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_a,
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_b) {
  GRANITE_ASSERT(bearings_a.size() == bearings_b.size());

  const size_t N = bearings_a.size();

  if (N < 4) {
    return {};
  }

  using MatrixX9d = Eigen::Matrix<double, Eigen::Dynamic, 9>;

  MatrixX9d A;
  A.setZero(2 * N, 9);

  for (size_t pt_idx = 0; pt_idx < N; pt_idx++) {
    const auto& b_a = bearings_a.at(pt_idx);
    const auto& b_b = bearings_b.at(pt_idx);

    // eq. (4.3)
    A(2 * pt_idx, 0) = 0.0;
    A(2 * pt_idx, 1) = 0.0;
    A(2 * pt_idx, 2) = 0.0;
    A(2 * pt_idx, 3) = -b_a.z() * b_b.x();
    A(2 * pt_idx, 4) = -b_a.z() * b_b.y();
    A(2 * pt_idx, 5) = -b_a.z() * b_b.z();
    A(2 * pt_idx, 6) = b_a.y() * b_b.x();
    A(2 * pt_idx, 7) = b_a.y() * b_b.y();
    A(2 * pt_idx, 8) = b_a.y() * b_b.z();

    A(2 * pt_idx + 1, 0) = b_a.z() * b_b.x();
    A(2 * pt_idx + 1, 1) = b_a.z() * b_b.y();
    A(2 * pt_idx + 1, 2) = b_a.z() * b_b.z();
    A(2 * pt_idx + 1, 3) = 0.0;
    A(2 * pt_idx + 1, 4) = 0.0;
    A(2 * pt_idx + 1, 5) = 0.0;
    A(2 * pt_idx + 1, 6) = -b_a.x() * b_b.x();
    A(2 * pt_idx + 1, 7) = -b_a.x() * b_b.y();
    A(2 * pt_idx + 1, 8) = -b_a.x() * b_b.z();
  }

  // algorithm 4.1 step (iii)
  const Eigen::JacobiSVD<MatrixX9d> homography_svd(A, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 1> h = homography_svd.matrixV().col(8);

  // eq. (4.2)
  return Eigen::Matrix3d(h.data()).transpose();
}

// decompose homography into 8 motion hypothesis
// method from "Motion and Structure from motion in a piecewise planar
// environment" - Faugeras, Lustman 1988
// covers only case 1
// might return empty vector
Eigen::aligned_vector<Sophus::SE3d> decompose_homography(
    const Eigen::Matrix3d& H) {
  const Eigen::JacobiSVD<Eigen::Matrix3d> homography_svd(
      H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const double s = homography_svd.matrixU().determinant() *
                   homography_svd.matrixV().determinant();
  const Eigen::Vector3d& sing_val = homography_svd.singularValues();

  // do not cover case 2 (motion along V * n) or 3 (no motion at all)
  if (sing_val(0) / sing_val(1) - 1.0 < 1e-5 ||
      sing_val(1) / sing_val(2) - 1.0 < 1e-5) {
    return {};
  }

  Eigen::aligned_vector<Sophus::SE3d> motion_hypothesis(8);

  // cover case 1
  // n' = (x1 0 x3)'
  // eq. (12)
  const double x1_abs =
      sqrt((sing_val(0) * sing_val(0) - sing_val(1) * sing_val(1)) /
           (sing_val(0) * sing_val(0) - sing_val(2) * sing_val(2)));
  const double x3_abs =
      sqrt((sing_val(1) * sing_val(1) - sing_val(2) * sing_val(2)) /
           (sing_val(0) * sing_val(0) - sing_val(2) * sing_val(2)));
  const double x1[] = {x1_abs, x1_abs, -x1_abs, -x1_abs};
  const double x3[] = {x3_abs, -x3_abs, x3_abs, -x3_abs};

  // subcase d' > 0
  // eq. (13)
  const double cos_theta =
      (sing_val(1) * sing_val(1) + sing_val(0) * sing_val(2)) /
      ((sing_val(0) + sing_val(2)) * sing_val(1));
  const double sin_theta_abs =
      sqrt((sing_val(0) * sing_val(0) - sing_val(1) * sing_val(1)) *
           (sing_val(1) * sing_val(1) - sing_val(2) * sing_val(2))) /
      ((sing_val(0) + sing_val(2)) * sing_val(1));
  const double sin_theta[] = {sin_theta_abs, -sin_theta_abs, -sin_theta_abs,
                              sin_theta_abs};

  for (size_t i = 0; i < 4; i++) {
    // eq. between (12) and (13)
    Eigen::Matrix3d R_prime = Eigen::Matrix3d::Identity();
    R_prime(0, 0) = cos_theta;
    R_prime(0, 2) = -sin_theta[i];
    R_prime(2, 0) = sin_theta[i];
    R_prime(2, 2) = cos_theta;

    // eq. (8)
    const Eigen::Matrix3d R = s * homography_svd.matrixU() * R_prime *
                              homography_svd.matrixV().transpose();

    // eq. (14)
    Eigen::Vector3d t_prime;
    t_prime.x() = x1[i];
    t_prime.y() = 0.0;
    t_prime.z() = -x3[i];
    t_prime *= sing_val(0) - sing_val(2);

    // eq. (8)
    const Eigen::Vector3d t = homography_svd.matrixU() * t_prime;

    motion_hypothesis[i] = Sophus::SE3d(R, t);
  }

  // subcase d' < 0
  // eq. (15)
  const double cos_phi =
      (sing_val(0) * sing_val(2) - sing_val(1) * sing_val(1)) /
      ((sing_val(0) - sing_val(2)) * sing_val(1));
  const double sin_phi_abs =
      sqrt((sing_val(0) * sing_val(0) - sing_val(1) * sing_val(1)) *
           (sing_val(1) * sing_val(1) - sing_val(2) * sing_val(2))) /
      ((sing_val(0) - sing_val(2)) * sing_val(1));
  const double sin_phi[] = {sin_phi_abs, -sin_phi_abs, -sin_phi_abs,
                            sin_phi_abs};

  for (size_t i = 0; i < 4; i++) {
    // eq. between (14) and (15)
    Eigen::Matrix3d R_prime = -Eigen::Matrix3d::Identity();
    R_prime(0, 0) = cos_phi;
    R_prime(0, 2) = sin_phi[i];
    R_prime(2, 0) = sin_phi[i];
    R_prime(2, 2) = -cos_phi;

    // eq. (8)
    const Eigen::Matrix3d R = s * homography_svd.matrixU() * R_prime *
                              homography_svd.matrixV().transpose();

    // eq. (16)
    Eigen::Vector3d t_prime;
    t_prime.x() = x1[i];
    t_prime.y() = 0.0;
    t_prime.z() = x3[i];
    t_prime *= sing_val(0) + sing_val(2);

    // eq. (8)
    const Eigen::Vector3d t = homography_svd.matrixU() * t_prime;

    motion_hypothesis[4 + i] = Sophus::SE3d(R, t);
  }

  return motion_hypothesis;
}

Eigen::aligned_vector<Sophus::SE3d> decompose_essential(
    const Eigen::Matrix3d& E) {
  const static Eigen::Matrix3d Rz_pi_half =
      (Eigen::Matrix3d() << 0, -1, 0, 1, 0, 0, 0, 0, 1).finished();

  const Eigen::JacobiSVD<Eigen::Matrix3d> essential_svd(
      E, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Vector3d sing_val = essential_svd.singularValues();

  const double scale = sing_val(0);

  const Eigen::Vector3d t_a = scale * essential_svd.matrixU().col(2);
  const Eigen::Vector3d t_b = -t_a;

  Eigen::Matrix3d R_a = essential_svd.matrixU() * Rz_pi_half.transpose() *
                        essential_svd.matrixV().transpose();
  if (R_a.determinant() < 0) R_a = -R_a;
  Eigen::Matrix3d R_b = essential_svd.matrixU() * Rz_pi_half *
                        essential_svd.matrixV().transpose();
  if (R_b.determinant() < 0) R_b = -R_b;

  Eigen::aligned_vector<Sophus::SE3d> motion_hypothesis(4);
  motion_hypothesis[0] = Sophus::SE3d(R_a, t_a);
  motion_hypothesis[1] = Sophus::SE3d(R_a, t_b);
  motion_hypothesis[2] = Sophus::SE3d(R_b, t_a);
  motion_hypothesis[3] = Sophus::SE3d(R_b, t_b);

  return motion_hypothesis;
}

Eigen::aligned_vector<Sophus::SE3d> filter_motion_hypothesis(
    const Eigen::aligned_vector<Sophus::SE3d>& motion_hypothesis,
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_a,
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_b) {
  Eigen::aligned_vector<Sophus::SE3d> filtered_motion_hypothesis;

  for (const auto& mot_hyp : motion_hypothesis) {
    bool exclusion_criteria = false;
    size_t small_parallax_count = 0;
    for (size_t bearing_idx = 0; bearing_idx < bearings_a.size();
         bearing_idx++) {
      const Eigen::Vector3d& bearing_a = bearings_a.at(bearing_idx);
      const Eigen::Vector3d bearing_b_rotated =
          mot_hyp.so3() * bearings_b.at(bearing_idx);

      // if all points have low parallax -> rotation only case
      // bearing vectors parallel?
      const double cos_angle = bearing_a.transpose() * bearing_b_rotated;

      if (std::abs(cos_angle) >= std::acos(0.5 * M_PI / 180.0)) {
        small_parallax_count++;
      } else {
        // check if bearing vectors point into same direction
        // but only if not parallel
        const Eigen::Vector3d bearing_cross =
            bearing_b_rotated.cross(bearing_a);
        const double dist_a =
            -mot_hyp.translation().cross(bearing_a).transpose() * bearing_cross;
        const double dist_b =
            -mot_hyp.translation().cross(bearing_b_rotated).transpose() *
            bearing_cross;

        if (dist_a <= 0.0 || dist_b <= 0.0) {
          exclusion_criteria = true;
          break;
        }
      }
    }
    if (!exclusion_criteria && small_parallax_count < bearings_a.size()) {
      filtered_motion_hypothesis.push_back(mot_hyp);
    }
  }

  return filtered_motion_hypothesis;
}

std::vector<double> compute_reprojection_errors(
    const Sophus::SE3d& T_cama_camb,
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_a,
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_b) {
  GRANITE_ASSERT(bearings_a.size() == bearings_b.size());
  std::vector<double> errors(bearings_a.size());

  for (size_t bearing_idx = 0; bearing_idx < bearings_a.size(); bearing_idx++) {
    const auto& bearing_a = bearings_a.at(bearing_idx);
    const auto& bearing_b = bearings_b.at(bearing_idx);

    double reprojection_error;

    const Eigen::Vector4d p_tri =
        BundleAdjustmentBase::triangulate(bearing_a, bearing_b, T_cama_camb);
    if (p_tri.allFinite() && p_tri[3] > 0) {
      const double cama_id = p_tri[3];
      const Eigen::Vector3d p_camb =
          T_cama_camb.inverse() * (bearing_a / cama_id);

      reprojection_error = 1.0 - (bearing_b.transpose() * p_camb.normalized());
    } else {
      reprojection_error = std::numeric_limits<double>::infinity();
      // std::cout << "has infinite error" << std::endl;
    }

    errors.at(bearing_idx) = reprojection_error;
  }

  return errors;
}

std::vector<int> compute_inlier_set(
    const std::vector<double>& reprojection_errors,
    const double reprojection_threshold) {
  std::vector<int> inlier_idxs;
  for (size_t bearing_idx = 0; bearing_idx < reprojection_errors.size();
       bearing_idx++) {
    if (std::isfinite(reprojection_errors.at(bearing_idx)) &&
        reprojection_errors.at(bearing_idx) <= reprojection_threshold) {
      inlier_idxs.push_back(bearing_idx);
    }
  }
  return inlier_idxs;
}

std::vector<int> compute_inlier_set(
    const Sophus::SE3d& T_cama_camb,
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_a,
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_b,
    const double reprojection_threshold) {
  const std::vector<double> reprojection_errors =
      compute_reprojection_errors(T_cama_camb, bearings_a, bearings_b);
  return compute_inlier_set(reprojection_errors, reprojection_threshold);
}

struct RansacResult {
  static constexpr double INF = std::numeric_limits<double>::infinity();
  double avg_error = RansacResult::INF;
  std::vector<int> inlier_idx;
  Sophus::SE3d T_a_b;

  RansacResult() = default;

  RansacResult(const RansacResult& other) = default;

  RansacResult(RansacResult&& other) noexcept
      : avg_error(std::exchange(other.avg_error, RansacResult::INF)),
        inlier_idx(other.inlier_idx),
        T_a_b(other.T_a_b) {}

  RansacResult& operator=(const RansacResult& other) = default;

  RansacResult& operator=(RansacResult&& other) noexcept {
    avg_error = std::exchange(other.avg_error, RansacResult::INF);
    inlier_idx = std::move(other.inlier_idx);
    T_a_b = other.T_a_b;

    return *this;
  }
};

class ParallelRANSACIteration {
  const std::vector<size_t>& indices;
  const Eigen::aligned_vector<Eigen::Vector3d>& bearings_a;
  const Eigen::aligned_vector<Eigen::Vector3d>& bearings_b;
  const double reprojection_threshold;

 public:
  const static size_t N = 5;
  RansacResult best_res;

  void operator()(const tbb::blocked_range<size_t>& r) {
    RansacResult tmp_best_res = best_res;
    auto gen = std::mt19937(std::random_device{}());

    for (size_t i = r.begin(); i != r.end(); ++i) {
      // sample N elements from bearing vectors
      std::vector<int> picked_indices(N);
      std::sample(indices.begin(), indices.end(), picked_indices.begin(), N,
                  gen);

      Eigen::aligned_vector<Eigen::Vector3d> picked_bearings_a(N),
          picked_bearings_b(N);
      for (size_t idx = 0; idx < N; idx++) {
        picked_bearings_a.at(idx) = bearings_a.at(picked_indices.at(idx));
        picked_bearings_b.at(idx) = bearings_b.at(picked_indices.at(idx));
      }

      // compute motion hypothesis via 5pt and homography algorithm
      Eigen::aligned_vector<Sophus::SE3d> motion_hypothesis;
      const auto H_opt =
          compute_homography(picked_bearings_a, picked_bearings_b);
      if (H_opt) {
        motion_hypothesis =
            filter_motion_hypothesis(decompose_homography(H_opt.value()),
                                     picked_bearings_a, picked_bearings_b);
      }
      const opengv::relative_pose::CentralRelativeAdapter opengv_adapter(
          picked_bearings_a, picked_bearings_b);
      const Eigen::aligned_vector<Eigen::Matrix3d> five_pt_essentials =
          opengv::relative_pose::fivept_nister(opengv_adapter);

      for (const auto& E : five_pt_essentials) {
        const Eigen::aligned_vector<Sophus::SE3d> mot_hyp =
            filter_motion_hypothesis(decompose_essential(E), picked_bearings_a,
                                     picked_bearings_b);

        motion_hypothesis.insert(std::end(motion_hypothesis),
                                 std::begin(mot_hyp), std::end(mot_hyp));
      }

      // find best motion hypothesis
      for (auto& mot_hyp : motion_hypothesis) {
        std::vector<double> reprojection_errors =
            compute_reprojection_errors(mot_hyp, bearings_a, bearings_b);

        const std::vector<int> inlier_idxs =
            compute_inlier_set(reprojection_errors, reprojection_threshold);

        double avg_error = 0;
        for (auto idx : inlier_idxs) {
          avg_error += reprojection_errors.at(idx);
        }
        avg_error /= inlier_idxs.size();

        // TODO address magic number
        if (inlier_idxs.size() > 20 &&
            inlier_idxs.size() > tmp_best_res.inlier_idx.size() &&
            avg_error <= tmp_best_res.avg_error) {
          tmp_best_res.avg_error = avg_error;
          tmp_best_res.inlier_idx = std::move(inlier_idxs);
          tmp_best_res.T_a_b = mot_hyp;
        }
      }
    }
    best_res = std::move(tmp_best_res);
  }

  ParallelRANSACIteration(ParallelRANSACIteration& other, tbb::split)
      : indices(other.indices),
        bearings_a(other.bearings_a),
        bearings_b(other.bearings_b),
        reprojection_threshold(other.reprojection_threshold),
        best_res(other.best_res) {}

  void join(const ParallelRANSACIteration& other) {
    if (other.best_res.inlier_idx.size() > best_res.inlier_idx.size() &&
        other.best_res.avg_error <= best_res.avg_error) {
      // TODO is std::move possible?
      best_res = other.best_res;
    }
  }

  ParallelRANSACIteration(
      const std::vector<size_t>& indices,
      const Eigen::aligned_vector<Eigen::Vector3d>& bearings_a,
      const Eigen::aligned_vector<Eigen::Vector3d>& bearings_b,
      const double reprojection_threshold)
      : indices(indices),
        bearings_a(bearings_a),
        bearings_b(bearings_b),
        reprojection_threshold(reprojection_threshold) {
    GRANITE_ASSERT(indices.size() >= N);
  }
};

std::optional<Sophus::SE3d> relative_pose(
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_a,
    const Eigen::aligned_vector<Eigen::Vector3d>& bearings_b,
    const double reprojection_threshold, unsigned int max_ransac_iter = 20) {
  GRANITE_ASSERT(bearings_a.size() == bearings_b.size());

  if (bearings_a.size() < ParallelRANSACIteration::N) {
    return {};
  }

  // RANSAC
  std::vector<size_t> indices(bearings_a.size());
  for (size_t idx = 0; idx < bearings_a.size(); idx++) {
    indices.at(idx) = idx;
  }

  auto PRI = ParallelRANSACIteration(indices, bearings_a, bearings_b,
                                     reprojection_threshold);
  tbb::parallel_reduce(tbb::blocked_range<size_t>(0, max_ransac_iter), PRI);
  const RansacResult& best_res = PRI.best_res;

  // prepare OpenGV Adapter
  opengv::relative_pose::CentralRelativeAdapter adapter(bearings_a, bearings_b);

  // prepare adapter for nonlinear refinement
  adapter.setR12(best_res.T_a_b.rotationMatrix());
  adapter.sett12(best_res.T_a_b.translation());

  /*const std::vector<int> current_inlier_idxs = compute_inlier_set(
      best_res.T_a_b, bearings_a, bearings_b, reprojection_threshold);*/

  const opengv::transformation_t T_cama_camb_opti_opengv =
      opengv::relative_pose::optimize_nonlinear(adapter, best_res.inlier_idx);

  const Sophus::SE3d T_cama_camb_opti =
      Sophus::SE3d(T_cama_camb_opti_opengv.topLeftCorner(3, 3),
                   T_cama_camb_opti_opengv.topRightCorner(3, 1));

  if (std::isfinite(best_res.avg_error)) {
    return T_cama_camb_opti;
  } else {
    return {};
  }
}

struct InitializationResult {
  double score = -1;
  double parallax = -1;
  size_t num_enough_parallax = 0;
  TimeCamId tcid_a;
  TimeCamId tcid_b;
  Eigen::aligned_vector<Eigen::Vector4d> pts_triangulated;
  std::vector<int> triangulated_idxs;
  Eigen::aligned_vector<Eigen::Vector2d> points_a_inlier;
  Eigen::aligned_vector<Eigen::Vector2d> points_b_inlier;
  Sophus::SE3d T_a_b;

  InitializationResult() = default;

  explicit InitializationResult(const TimeCamId& tcid_a,
                                const TimeCamId& tcid_b)
      : tcid_a(tcid_a), tcid_b(tcid_b) {}

  InitializationResult(const InitializationResult& other) = default;

  InitializationResult(InitializationResult&& other) noexcept
      : score(std::exchange(other.score, -1)),
        parallax(std::exchange(other.parallax, -1)),
        num_enough_parallax(std::exchange(other.num_enough_parallax, 0)),
        tcid_a(std::exchange(other.tcid_a, TimeCamId())),
        tcid_b(std::exchange(other.tcid_b, TimeCamId())),
        pts_triangulated(std::move(other.pts_triangulated)),
        triangulated_idxs(std::move(other.triangulated_idxs)),
        points_a_inlier(std::move(other.points_a_inlier)),
        points_b_inlier(std::move(other.points_b_inlier)),
        T_a_b(other.T_a_b) {}

  InitializationResult& operator=(const InitializationResult& other) = default;

  InitializationResult& operator=(InitializationResult&& other) noexcept {
    score = std::exchange(other.score, -1);
    parallax = std::exchange(other.parallax, -1);
    num_enough_parallax = std::exchange(other.num_enough_parallax, 0);
    tcid_a = std::exchange(other.tcid_a, TimeCamId());
    tcid_b = std::exchange(other.tcid_b, TimeCamId());
    pts_triangulated = std::move(other.pts_triangulated);
    triangulated_idxs = std::move(other.triangulated_idxs);
    points_a_inlier = std::move(other.points_a_inlier);
    points_b_inlier = std::move(other.points_b_inlier);
    T_a_b = other.T_a_b;

    return *this;
  }
};

InitializationResult evaluate_initialization_candidate(
    const TimeCamId& tcid_a, const TimeCamId& tcid_b,
    const OpticalFlowResult::Ptr& opt_flow_a,
    const OpticalFlowResult::Ptr& opt_flow_b, const Sophus::SE3d& current_T_a_b,
    const GenericCamera<double>& cam_a, const GenericCamera<double>& cam_b,
    double min_parallax_deg = 5.0, unsigned int max_ransac_iter = 20) {
  // look for correspondences in both optical flow results
  // store the bearing vectors for relative pose estimation
  std::vector<KeypointId> obs_idxs;
  Eigen::aligned_vector<Eigen::Vector2d> points_a, points_b;
  Eigen::aligned_vector<Eigen::Vector3d> bearing_vectors_a, bearing_vectors_b;
  for (const auto& kv_new_obs : opt_flow_a->observations.at(0)) {
    auto it = opt_flow_b->observations.at(0).find(kv_new_obs.first);
    if (it != opt_flow_b->observations.at(0).end()) {
      const Eigen::Vector2d p_a =
          kv_new_obs.second.translation().cast<double>();
      const Eigen::Vector2d p_b = it->second.translation().cast<double>();

      Eigen::Vector4d p_a_3d, p_b_3d;
      bool valid_a = cam_a.unproject(p_a, p_a_3d);
      bool valid_b = cam_b.unproject(p_b, p_b_3d);
      if (valid_a && valid_b) {
        obs_idxs.push_back(kv_new_obs.first);
        points_a.push_back(p_a);
        points_b.push_back(p_b);
        bearing_vectors_a.push_back(p_a_3d.head<3>());
        bearing_vectors_b.push_back(p_b_3d.head<3>());
        // TODO
        // opt_flow_a->pyramid_levels.at(0).at(kv_new_obs.first);
      }
    }
  }

  // TODO address magic number
  const double reprojection_threshold =
      1.0 - cos(atan(sqrt(3.0) * 1.0 / cam_b.getParam()[0]));

  const std::optional<Sophus::SE3d> T_a_b =
      relative_pose(bearing_vectors_a, bearing_vectors_b,
                    reprojection_threshold, max_ransac_iter);

  if (!T_a_b.has_value()) {
    return InitializationResult(tcid_a, tcid_b);
  }

  // check if it is better than the current (rotation only) estimate

  std::vector<double> current_reprojection_errors = compute_reprojection_errors(
      current_T_a_b, bearing_vectors_a, bearing_vectors_b);
  const std::vector<int> current_inlier_idxs =
      compute_inlier_set(current_reprojection_errors, reprojection_threshold);

  double current_avg_error = 0;
  for (auto idx : current_inlier_idxs) {
    current_avg_error += current_reprojection_errors.at(idx);
  }
  current_avg_error /= current_inlier_idxs.size();

  std::vector<double> new_reprojection_errors = compute_reprojection_errors(
      T_a_b.value(), bearing_vectors_a, bearing_vectors_b);
  const std::vector<int> new_inlier_idxs =
      compute_inlier_set(new_reprojection_errors, reprojection_threshold);

  double new_avg_error = 0;
  for (auto idx : new_inlier_idxs) {
    new_avg_error += new_reprojection_errors.at(idx);
  }
  new_avg_error /= new_inlier_idxs.size();

  if (new_inlier_idxs.size() >= current_inlier_idxs.size() &&
      new_avg_error <= current_avg_error) {
    InitializationResult result(tcid_a, tcid_b);

    // triangulate points
    for (const auto inlier_idx : new_inlier_idxs) {
      const Eigen::Vector4d p_triangulated = BundleAdjustmentBase::triangulate(
          bearing_vectors_a.at(inlier_idx), bearing_vectors_b.at(inlier_idx),
          T_a_b.value());

      if (p_triangulated.allFinite() && p_triangulated[3] > 0) {
        result.pts_triangulated.push_back(p_triangulated);
        result.triangulated_idxs.push_back(obs_idxs.at(inlier_idx));
        result.points_a_inlier.push_back(points_a.at(inlier_idx));
        result.points_b_inlier.push_back(points_b.at(inlier_idx));
      }

      double angle =
          std::acos(bearing_vectors_a.at(inlier_idx).transpose() *
                    (T_a_b.value().so3() * bearing_vectors_b.at(inlier_idx)));

      if (angle >= min_parallax_deg * M_PI / 180.0) {
        result.num_enough_parallax++;
      }
    }

    // TODO address magic number
    if (result.pts_triangulated.size() < 20) {
      return InitializationResult(tcid_a, tcid_b);
    }

    // normalize to a depth of avg_d_should
    constexpr double avg_d_should = 2.0;
    // calc average depth of triangulated points
    double avg_d = 0;
    for (const auto& p_triangulated : result.pts_triangulated) {
      avg_d += 1.0 / p_triangulated(3);
    }
    avg_d /= result.pts_triangulated.size();
    const double scale_factor = avg_d_should / avg_d;

    // normalize to avg_d_should
    for (auto& pts : result.pts_triangulated) {
      pts(3) /= scale_factor;
    }
    result.T_a_b = Sophus::SE3d(T_a_b.value().so3(),
                                T_a_b.value().translation() * scale_factor);

    // compute parallax
    result.parallax =
        2.0 * atan(result.T_a_b.translation().norm() / (2.0 * avg_d_should));

    std::vector<double> reprojection_errors = compute_reprojection_errors(
        result.T_a_b, bearing_vectors_a, bearing_vectors_b);

    double error_sum = std::accumulate(reprojection_errors.begin(),
                                       reprojection_errors.end(), 0.0);

    result.score =
        double(bearing_vectors_a.size() * bearing_vectors_a.size()) / error_sum;

    return result;
  }

  return InitializationResult(tcid_a, tcid_b);
}

class ParallelInitializationCandidateEvaluation {
  const OpticalFlowResult::Ptr& opt_flow_a;
  const Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr>& opt_flow_bs;
  const Eigen::aligned_map<int64_t, Sophus::SE3d>& current_T_a_bs;
  const GenericCamera<double>& cam_a;
  const GenericCamera<double>& cam_b;
  unsigned int max_ransac_iter;
  double min_parallax;

 public:
  InitializationResult best_init_res;

  void operator()(const tbb::blocked_range<size_t>& r) {
    InitializationResult tmp_best_init_res = best_init_res;
    const TimeCamId tcid_a(opt_flow_a->t_ns, 0);
    for (size_t i = r.begin(); i != r.end(); ++i) {
      auto opt_flow_iter = std::next(opt_flow_bs.begin(), i);
      // TODO make cam_id parameter
      if (opt_flow_iter->first != opt_flow_a->t_ns) {
        const TimeCamId tcid_b(opt_flow_iter->first, 0);
        InitializationResult init_res = evaluate_initialization_candidate(
            tcid_a, tcid_b, opt_flow_a, opt_flow_iter->second,
            current_T_a_bs.at(opt_flow_iter->first), cam_a, cam_b, min_parallax,
            max_ransac_iter);

        // TODO address magic number
        if (/*init_res.parallax > min_parallax * M_PI / 180.0 &&*/
            init_res.num_enough_parallax >= 20 &&
            init_res.score > tmp_best_init_res.score) {
          tmp_best_init_res = std::move(init_res);
        }
      }
    }
    best_init_res = std::move(tmp_best_init_res);
  }

  ParallelInitializationCandidateEvaluation(
      ParallelInitializationCandidateEvaluation& other, tbb::split)
      : opt_flow_a(other.opt_flow_a),
        opt_flow_bs(other.opt_flow_bs),
        current_T_a_bs(other.current_T_a_bs),
        cam_a(other.cam_a),
        cam_b(other.cam_b),
        max_ransac_iter(other.max_ransac_iter),
        min_parallax(other.min_parallax),
        best_init_res(other.best_init_res) {}

  void join(const ParallelInitializationCandidateEvaluation& other) {
    if (other.best_init_res.score > best_init_res.score) {
      // TODO is std::move possible?
      best_init_res = other.best_init_res;
    }
  }

  ParallelInitializationCandidateEvaluation(
      const OpticalFlowResult::Ptr& opt_flow_a,
      const Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr>& opt_flow_bs,
      const Eigen::aligned_map<int64_t, Sophus::SE3d>& current_T_a_bs,
      const GenericCamera<double>& cam_a, const GenericCamera<double>& cam_b,
      unsigned int max_ransac_iter, double min_parallax)
      : opt_flow_a(opt_flow_a),
        opt_flow_bs(opt_flow_bs),
        current_T_a_bs(current_T_a_bs),
        cam_a(cam_a),
        cam_b(cam_b),
        max_ransac_iter(max_ransac_iter),
        min_parallax(min_parallax) {}
};

}  // namespace granite