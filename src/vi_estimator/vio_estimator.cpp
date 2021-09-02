/**
MIT License

This file is part of the Granite project which is based on Basalt.
https://github.com/DLR-RM/granite

Copyright (c) Martin Wudenka, Deutsches Zentrum für Luft- und Raumfahrt

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

/**
Original license of Basalt:
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <granite/vi_estimator/vio_estimator.h>

#include <granite/vi_estimator/keypoint_vio.h>
#include <granite/vi_estimator/keypoint_vo.h>

namespace granite {

VioEstimatorBase::Ptr VioEstimatorFactory::getVioEstimator(
    const VioConfig& config, const Calibration<double>& cam,
    const Eigen::Vector3d& g, bool use_imu) {
  VioEstimatorBase::Ptr res;

  if (use_imu) {
    res.reset(new KeypointVioEstimator(g, cam, config));
  } else {
    res.reset(new KeypointVoEstimator(cam, config));
  }

  return res;
}

std::pair<double, double> alignSVD(
    const std::vector<int64_t>& filter_t_ns,
    const Eigen::aligned_vector<Eigen::Vector3d>& filter_t_w_i,
    const std::vector<int64_t>& gt_t_ns,
    const Eigen::aligned_vector<Eigen::Vector3d>& gt_t_w_i,
    Sophus::SE3d& T_gt_est, Sophus::Sim3d& sT_gt_est, bool verbose) {
  Eigen::aligned_vector<Eigen::Vector3d> est_associations;
  Eigen::aligned_vector<Eigen::Vector3d> gt_associations;

  for (size_t i = 0; i < filter_t_w_i.size(); i++) {
    int64_t t_ns = filter_t_ns[i];

    size_t j;
    for (j = 0; j < gt_t_ns.size(); j++) {
      if (gt_t_ns.at(j) > t_ns) break;
    }
    j--;

    if (j >= gt_t_ns.size() - 1) {
      continue;
    }

    double dt_ns = t_ns - gt_t_ns.at(j);
    double int_t_ns = gt_t_ns.at(j + 1) - gt_t_ns.at(j);

    GRANITE_ASSERT_STREAM(dt_ns >= 0, "dt_ns " << dt_ns);
    GRANITE_ASSERT_STREAM(int_t_ns > 0, "int_t_ns " << int_t_ns);

    // Skip if the interval between gt larger than 100ms
    // std::cout << int_t_ns << std::endl;
    // if (int_t_ns > 1.1e8) continue;
    // Skip if t_ns is further away than 50ms from a gt point
    if (dt_ns >= 0.05e9 && double(gt_t_ns.at(j + 1) - t_ns) >= 0.05e9) continue;

    double ratio = dt_ns / int_t_ns;

    GRANITE_ASSERT(ratio >= 0);
    GRANITE_ASSERT(ratio < 1);

    Eigen::Vector3d gt = (1 - ratio) * gt_t_w_i[j] + ratio * gt_t_w_i[j + 1];

    gt_associations.emplace_back(gt);
    est_associations.emplace_back(filter_t_w_i[i]);
  }

  const int num_kfs = est_associations.size();

  if (num_kfs < 3) return std::make_pair(-1.0, -1.0);

  Eigen::Matrix<double, 3, Eigen::Dynamic> gt, est;
  gt.setZero(3, num_kfs);
  est.setZero(3, num_kfs);

  for (size_t i = 0; i < est_associations.size(); i++) {
    gt.col(i) = gt_associations[i];
    est.col(i) = est_associations[i];
  }

  Eigen::Vector3d mean_gt = gt.rowwise().mean();
  Eigen::Vector3d mean_est = est.rowwise().mean();

  gt.colwise() -= mean_gt;
  est.colwise() -= mean_est;

  Eigen::Matrix3d cov = gt * est.transpose();

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d S;
  S.setIdentity();

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
    S(2, 2) = -1;

  Eigen::Matrix3d rot_gt_est = svd.matrixU() * S * svd.matrixV().transpose();
  Eigen::Vector3d trans = mean_gt - rot_gt_est * mean_est;

  T_gt_est = Sophus::SE3d(rot_gt_est, trans);

  // estimate scale
  double dots = 0.0;
  double squared_norms = 0.0;
  for (size_t i = 0; i < est_associations.size(); i++) {
    Eigen::Vector3d est_rot_t_w_i =
        rot_gt_est * (est_associations.at(i) - mean_est);
    dots += est_rot_t_w_i.transpose() * (gt_associations.at(i) - mean_gt);
    squared_norms += est_rot_t_w_i.squaredNorm();
  }
  double scale = dots / squared_norms;

  if (scale < Sophus::Constants<double>::epsilon()) {
    scale = 1;
  }

  Eigen::Matrix3d scaled_rot_gt_est = scale * rot_gt_est;
  Eigen::Vector3d scaled_trans = mean_gt - scaled_rot_gt_est * mean_est;

  sT_gt_est = Sophus::Sim3d(Sophus::RxSO3d(scaled_rot_gt_est), scaled_trans);

  double se3_error = 0;
  for (size_t i = 0; i < est_associations.size(); i++) {
    const Eigen::Vector3d se3_est = T_gt_est * est_associations[i];
    Eigen::Vector3d res = se3_est - gt_associations[i];

    se3_error += res.transpose() * res;
  }

  se3_error /= est_associations.size();
  se3_error = std::sqrt(se3_error);

  double sim3_error = 0;
  for (size_t i = 0; i < est_associations.size(); i++) {
    const Eigen::Vector3d sim3_est = sT_gt_est * est_associations[i];
    Eigen::Vector3d res = sim3_est - gt_associations[i];

    sim3_error += res.transpose() * res;
  }

  sim3_error /= est_associations.size();
  sim3_error = std::sqrt(sim3_error);

  if (verbose) {
    std::cout << "SE3 T_gt_est:\n" << T_gt_est.matrix() << std::endl;
    std::cout << "Sim3 T_gt_est:\n" << sT_gt_est.matrix() << std::endl;
    std::cout << "SE3 RMS ATE: " << se3_error << std::endl;
    std::cout << "Sim3 RMS ATE: " << sim3_error << std::endl;
    std::cout << "number of associations: " << num_kfs << std::endl;
  }

  return std::make_pair(se3_error, sim3_error);
}
}  // namespace granite
