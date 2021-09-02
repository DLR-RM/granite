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

#include <atomic>

#include <granite/optical_flow/optical_flow.h>
#include <granite/utils/imu_types.h>

namespace granite {

struct VioStateData {
  typedef std::shared_ptr<VioStateData> Ptr;

  PoseVelBiasState state;
  size_t map_idx;

  double scale_variance = std::numeric_limits<double>::quiet_NaN();
  double drift_variance = std::numeric_limits<double>::quiet_NaN();

  std::vector<FrameId> states_t_ns;
  Eigen::aligned_vector<Sophus::SE3d> states;
  std::vector<FrameId> frames_t_ns;
  Eigen::aligned_vector<Sophus::SE3d> frames;

  AbsOrderMap order;
  Eigen::MatrixXd H;
  Eigen::VectorXd b;
};

struct VioVisualizationData {
  typedef std::shared_ptr<VioVisualizationData> Ptr;

  FrameId t_ns;

  size_t map_idx = 0;

  std::vector<FrameId> states_t_ns;
  Eigen::aligned_vector<Sophus::SE3d> states;
  std::vector<FrameId> frames_t_ns;
  Eigen::aligned_vector<Sophus::SE3d> frames;

  Eigen::aligned_vector<Eigen::Vector3d> points;
  std::vector<int> point_ids;

  OpticalFlowResult::Ptr opt_flow_res;

  double negative_entropy_last_frame = 0;
  double average_negative_entropy_last_frame = 0;

  bool take_kf = false;

  std::vector<Eigen::aligned_vector<Eigen::Vector4d>> projections;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class VioEstimatorBase {
 public:
  typedef std::shared_ptr<VioEstimatorBase> Ptr;

  VioEstimatorBase()
      : out_state_queue(nullptr),
        out_marg_queue(nullptr),
        out_vis_queue(nullptr) {
    finished = false;
  }

  bool isFinished() const { return finished; }

  int64_t getLastProcessedt_ns() { return last_processed_t_ns; }

  tbb::concurrent_bounded_queue<OpticalFlowResult::Ptr>* vision_data_queue =
      nullptr;
  tbb::concurrent_bounded_queue<ImuData::Ptr>* imu_data_queue = nullptr;

  tbb::concurrent_bounded_queue<VioStateData::Ptr>* out_state_queue = nullptr;
  tbb::concurrent_bounded_queue<MargData::Ptr>* out_marg_queue = nullptr;
  tbb::concurrent_bounded_queue<VioVisualizationData::Ptr>* out_vis_queue =
      nullptr;

  virtual void initialize(int64_t t_ns, const Sophus::SE3d& T_w_i,
                          const Eigen::Vector3d& vel_w_i,
                          const Eigen::Vector3d& bg,
                          const Eigen::Vector3d& ba) = 0;

  virtual void initialize(const Eigen::Vector3d& bg,
                          const Eigen::Vector3d& ba) = 0;

  virtual const Sophus::SE3d& getT_w_i_init() = 0;

  virtual inline void quit() { should_quit = true; }

  virtual void join() = 0;

  enum class TrackingState { UNINITIALIZED, TRACKING, LOST };

  std::atomic<TrackingState> tracking_state{TrackingState::UNINITIALIZED};

 protected:
  virtual void reset() {
    this_state_t_ns = -1;
    prev_state_t_ns = -1;
    last_processed_t_ns = -1;
  }

  FrameId this_state_t_ns = -1;
  FrameId prev_state_t_ns = -1;
  std::atomic<FrameId> last_processed_t_ns = -1;
  std::atomic<bool> should_quit = false;
  std::atomic<bool> finished;
  size_t map_idx = 0;
};

class VioEstimatorFactory {
 public:
  static VioEstimatorBase::Ptr getVioEstimator(const VioConfig& config,
                                               const Calibration<double>& cam,
                                               const Eigen::Vector3d& g,
                                               bool use_imu);
};

std::pair<double, double> alignSVD(
    const std::vector<int64_t>& filter_t_ns,
    const Eigen::aligned_vector<Eigen::Vector3d>& filter_t_w_i,
    const std::vector<int64_t>& gt_t_ns,
    const Eigen::aligned_vector<Eigen::Vector3d>& gt_t_w_i,
    Sophus::SE3d& T_gt_est, Sophus::Sim3d& sT_gt_est, bool verbose = true);
}  // namespace granite
