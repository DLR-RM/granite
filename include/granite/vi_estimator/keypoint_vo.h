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
#pragma once

#include <memory>
#include <thread>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <granite/imu/preintegration.h>
#include <granite/io/dataset_io.h>
#include <granite/utils/assert.h>
#include <granite/utils/test_utils.h>
#include <granite/camera/generic_camera.hpp>
#include <granite/camera/stereographic_param.hpp>
#include <granite/utils/sophus_utils.hpp>

#include <granite/vi_estimator/ba_base.h>
#include <granite/vi_estimator/vio_estimator.h>

namespace granite {

class KeypointVoEstimator : public VioEstimatorBase,
                            public BundleAdjustmentBase {
 public:
  typedef std::shared_ptr<KeypointVoEstimator> Ptr;

  static const int N = 9;
  typedef Eigen::Matrix<double, N, 1> VecN;
  typedef Eigen::Matrix<double, N, N> MatNN;
  typedef Eigen::Matrix<double, N, 3> MatN3;

  KeypointVoEstimator(const granite::Calibration<double>& calib,
                      const VioConfig& config);

  void initialize(FrameId t_ns, const Sophus::SE3d& T_w_i,
                  const Eigen::Vector3d& vel_w_i, const Eigen::Vector3d& bg,
                  const Eigen::Vector3d& ba) override;

  void initialize(const Eigen::Vector3d& bg,
                  const Eigen::Vector3d& ba) override;

  virtual ~KeypointVoEstimator() { join(); }

  void addIMUToQueue(const ImuData::Ptr& data);
  void addVisionToQueue(const OpticalFlowResult::Ptr& data);

 protected:
  virtual void reset() override;

 private:
  void init_first_pose(const FrameId t_ns, const Sophus::SE3d& T_w_i);

  void puplishData(const OpticalFlowResult::Ptr& opt_flow_meas) const;

  bool processFrame(const OpticalFlowResult::Ptr& opt_flow_meas);

  void rel_pose_initialisation(std::map<int64_t, int>& num_points_connected);

  std::map<FrameId, int> measure(const OpticalFlowResult::Ptr& data);

  // int64_t propagate();
  // void addNewState(int64_t data_t_ns);

  void marginalize(std::map<FrameId, int>& num_points_connected);

  void optimize(std::map<FrameId, int>& num_points_connected,
                int connected_finite, const bool filter = true);

  std::optional<double> stereoScaleUncertainty() const;

  std::optional<double> monoScaleDriftVariance() const;

  void create_keyframe(std::map<FrameId, int>& num_points_connected,
                       int connected_finite);

  void checkMargNullspace() const;

  void computeProjections(
      std::vector<Eigen::aligned_vector<Eigen::Vector4d>>& res) const;

  inline void increment_average_negative_entropy(double negative_entropy) {
    average_negative_entropy_last_frame +=
        (negative_entropy - average_negative_entropy_last_frame) /
        frames_after_kf;
  }

 public:
  int64_t get_t_ns() const {
    return frame_states.at(this_state_t_ns).getState().t_ns;
  }
  const Sophus::SE3d& get_T_w_i() const {
    return frame_states.at(this_state_t_ns).getState().T_w_i;
  }
  const Eigen::Vector3d& get_vel_w_i() const {
    return frame_states.at(this_state_t_ns).getState().vel_w_i;
  }

  const PoseVelBiasState& get_state() const {
    return frame_states.at(this_state_t_ns).getState();
  }
  PoseVelBiasState get_state(int64_t t_ns) const {
    PoseVelBiasState state;

    auto it = frame_states.find(t_ns);

    if (it != frame_states.end()) {
      return it->second.getState();
    }

    auto it2 = frame_poses.find(t_ns);
    if (it2 != frame_poses.end()) {
      state.T_w_i = it2->second.getPose();
    }

    return state;
  }
  // const MatNN get_cov() const { return cov.bottomRightCorner<N, N>(); }

  inline void setMaxStates(size_t val) { max_states = val; }
  inline void setMaxKfs(size_t val) { max_kfs = val; }

  Eigen::aligned_vector<Sophus::SE3d> getFrameStates() const {
    Eigen::aligned_vector<Sophus::SE3d> res;

    for (const auto& kv : frame_states) {
      res.push_back(kv.second.getState().T_w_i);
    }

    return res;
  }

  Eigen::aligned_vector<Sophus::SE3d> getFramePoses() const {
    Eigen::aligned_vector<Sophus::SE3d> res;

    for (const auto& kv : frame_poses) {
      res.push_back(kv.second.getPose());
    }

    return res;
  }

  Eigen::aligned_map<int64_t, Sophus::SE3d> getAllPosesMap() const {
    Eigen::aligned_map<int64_t, Sophus::SE3d> res;

    for (const auto& kv : frame_poses) {
      res[kv.first] = kv.second.getPose();
    }

    for (const auto& kv : frame_states) {
      res[kv.first] = kv.second.getState().T_w_i;
    }

    return res;
  }

  const Sophus::SE3d& getT_w_i_init() override { return T_w_i_init; }

  void join() override {
    if (processing_thread->joinable()) processing_thread->join();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  bool take_kf;
  int frames_after_kf;
  int num_processed_frames = 0;

  // Input

  Eigen::aligned_map<int64_t, OpticalFlowResult::Ptr> prev_opt_flow_res;

  // Marginalization
  AbsOrderMap marg_order;
  Eigen::MatrixXd marg_H;
  Eigen::VectorXd marg_b;

  // Optimization report
  AbsOrderMap optim_order;
  Eigen::MatrixXd optim_H;
  Eigen::VectorXd optim_b;

  size_t max_states;
  size_t max_kfs;

  Sophus::SE3d T_w_i_init;

  bool map_initialized = false;

  double lambda, min_lambda, max_lambda, lambda_vee;

  std::shared_ptr<std::thread> processing_thread;

  double negative_entropy_last_frame = 0;
  double average_negative_entropy_last_frame = 0;

  std::unordered_map<CamId, std::unordered_set<int>> this_unconnected_obs,
      this_untriangulated_obs, last_unconnected_obs, last_untriangulated_obs;
};
}  // namespace granite
