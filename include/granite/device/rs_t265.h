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

Copyright (c) 2019, Vladyslav Usenko, Michael Loipführer and Nikolaus Demmel.
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

#include <math.h>
#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>

#include <librealsense2/rs.hpp>

#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <tbb/concurrent_queue.h>

#include <granite/imu/imu_types.h>
#include <granite/optical_flow/optical_flow.h>
#include <granite/calibration/calibration.hpp>

namespace granite {

struct RsIMUData {
  double timestamp;
  Eigen::Vector3d data;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct RsPoseData {
  int64_t t_ns;
  Sophus::SE3d data;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class RsT265Device {
 public:
  using Ptr = std::shared_ptr<RsT265Device>;

  static constexpr int IMU_RATE = 200;
  static constexpr int NUM_CAMS = 2;

  RsT265Device(bool manual_exposure, int skip_frames, int webp_quality,
               double exposure_value = 10.0);
  ~RsT265Device();
  void start();
  void stop();

  bool setExposure(double exposure);  // in milliseconds
  void setSkipFrames(int skip);
  void setWebpQuality(int quality);

  std::shared_ptr<granite::Calibration<double>> exportCalibration();

  OpticalFlowInput::Ptr last_img_data;
  tbb::concurrent_bounded_queue<OpticalFlowInput::Ptr>* image_data_queue =
      nullptr;
  tbb::concurrent_bounded_queue<ImuData::Ptr>* imu_data_queue = nullptr;
  tbb::concurrent_bounded_queue<RsPoseData>* pose_data_queue = nullptr;

 private:
  bool manual_exposure;
  int skip_frames;
  int webp_quality;

  int frame_counter = 0;

  Eigen::aligned_deque<RsIMUData> gyro_data_queue;
  std::shared_ptr<RsIMUData> prev_accel_data;

  std::shared_ptr<granite::Calibration<double>> calib;

  rs2::context context;
  rs2::config config;
  rs2::pipeline pipe;
  rs2::sensor sensor;

  rs2::pipeline_profile profile;
};

}  // namespace granite
