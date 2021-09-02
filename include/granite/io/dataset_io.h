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

#include <array>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/bitset.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/set.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

#include <Eigen/Dense>
#include <granite/utils/sophus_utils.hpp>

#include <granite/image/image.h>
#include <granite/utils/assert.h>

#include <granite/utils/common_types.h>
#include <granite/camera/generic_camera.hpp>
#include <granite/camera/stereographic_param.hpp>

namespace granite {

struct ImageData {
  ImageData() : exposure(0) {}

  ManagedImage<PixelType>::Ptr img;
  double exposure;
};

struct Observations {
  Eigen::aligned_vector<Eigen::Vector2d> pos;
  std::vector<int> id;
};

struct GyroData {
  int64_t timestamp_ns;
  Eigen::Vector3d data;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct AccelData {
  int64_t timestamp_ns;
  Eigen::Vector3d data;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct PoseData {
  int64_t timestamp_ns;
  Sophus::SE3d data;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MocapPoseData {
  int64_t timestamp_ns;
  Sophus::SE3d data;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct AprilgridCornersData {
  int64_t timestamp_ns;
  int cam_id;

  Eigen::aligned_vector<Eigen::Vector2d> corner_pos;
  std::vector<int> corner_id;
};

class VioDataset {
 public:
  virtual ~VioDataset(){};

  virtual size_t get_num_cams() const = 0;

  virtual std::vector<int64_t> &get_image_timestamps() = 0;

  virtual const Eigen::aligned_vector<AccelData> &get_accel_data() const = 0;
  virtual const Eigen::aligned_vector<GyroData> &get_gyro_data() const = 0;
  virtual const std::vector<int64_t> &get_gt_timestamps() const = 0;
  virtual const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data()
      const = 0;
  virtual const Eigen::aligned_vector<Eigen::Vector3d> &get_gt_velocities()
      const = 0;
  virtual int64_t get_mocap_to_imu_offset_ns() const = 0;
  virtual std::vector<ImageData> get_image_data(int64_t t_ns) = 0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::shared_ptr<VioDataset> VioDatasetPtr;

class DatasetIoInterface {
 public:
  virtual bool read(const std::string &path, size_t num_cams = 2) = 0;
  virtual void reset() = 0;
  virtual VioDatasetPtr get_data() = 0;

  virtual ~DatasetIoInterface(){};
};

typedef std::shared_ptr<DatasetIoInterface> DatasetIoInterfacePtr;

class DatasetIoFactory {
 public:
  static DatasetIoInterfacePtr getDatasetIo(const std::string &dataset_type,
                                            bool load_mocap_as_gt = false);
};

inline void convert_copy_gray_image(const size_t width, const size_t height,
                             const uint8_t *data_in, uint16_t *data_out) {
  const size_t full_size = width * height;
  for (size_t i = 0; i < full_size; i++) {
    uint16_t val = data_in[i];
    val = val << 8;
    data_out[i] = val;
  }
}

inline void convert_copy_gray_image(const size_t width, const size_t height,
                             const uint16_t *data_in, uint8_t *data_out) {
  const size_t full_size = width * height;
  for (size_t i = 0; i < full_size; i++) {
    uint16_t val = data_in[i];
    val = val >> 8;
    data_out[i] = val;
  }
}

inline void convert_copy_gray_image(const size_t width, const size_t height,
                             const uint16_t *data_in, uint16_t *data_out) {
  const size_t full_size = width * height;
  std::memcpy(data_out, data_in, full_size * sizeof(uint16_t));
}

inline void convert_copy_gray_image(const size_t width, const size_t height,
                             const uint8_t *data_in, uint8_t *data_out) {
  const size_t full_size = width * height;
  std::memcpy(data_out, data_in, full_size * sizeof(uint8_t));
}

inline void convert_copy_color_image(const size_t width, const size_t height,
                              const uint16_t *data_in, uint8_t *data_out) {
  const size_t full_size = width * height;
  for (size_t i = 0; i < full_size; i++) {
    uint16_t val = data_in[i * 3];
    val = val >> 8;
    data_out[i] = val;
  }
}

inline void convert_copy_color_image(const size_t width, const size_t height,
                              const uint8_t *data_in, uint16_t *data_out) {
  const size_t full_size = width * height;
  for (size_t i = 0; i < full_size; i++) {
    uint16_t val = data_in[i * 3];
    val = val << 8;
    data_out[i] = val;
  }
}

inline void convert_copy_color_image(const size_t width, const size_t height,
                              const uint8_t *data_in, uint8_t *data_out) {
  const size_t full_size = width * height;
  for (size_t i = 0; i < full_size; i++) {
    uint16_t val = data_in[i * 3];
    data_out[i] = val;
  }
}

inline void convert_copy_color_image(const size_t width, const size_t height,
                              const uint16_t *data_in, uint16_t *data_out) {
  const size_t full_size = width * height;
  for (size_t i = 0; i < full_size; i++) {
    uint16_t val = data_in[i * 3];
    data_out[i] = val;
  }
}

}  // namespace granite

namespace cereal {

template <class Archive>
void serialize(Archive &ar, granite::GyroData &c) {
  ar(c.timestamp_ns, c.data);
}

template <class Archive>
void serialize(Archive &ar, granite::AccelData &c) {
  ar(c.timestamp_ns, c.data);
}

}  // namespace cereal
