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

#include <granite/io/dataset_io.h>
#include <granite/utils/common_types.h>
#include <granite/utils/exceptions.h>
#include <granite/utils/filesystem.h>

#include <opencv2/highgui/highgui.hpp>

namespace granite {

class KittiVioDataset : public VioDataset {
  size_t num_cams;

  std::string path;

  std::vector<int64_t> image_timestamps;
  std::unordered_map<int64_t, std::string> image_path;

  // vector of images for every timestamp
  // assumes vectors size is num_cams for every timestamp with null pointers for
  // missing frames
  // std::unordered_map<int64_t, std::vector<ImageData>> image_data;

  Eigen::aligned_vector<AccelData> accel_data;
  Eigen::aligned_vector<GyroData> gyro_data;

  std::vector<int64_t> gt_timestamps;  // ordered gt timestamps
  Eigen::aligned_vector<Sophus::SE3d>
      gt_pose_data;  // TODO: change to eigen aligned

  int64_t mocap_to_imu_offset_ns;

 public:
  ~KittiVioDataset(){};

  size_t get_num_cams() const { return num_cams; }

  std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }

  const Eigen::aligned_vector<AccelData> &get_accel_data() const {
    return accel_data;
  }
  const Eigen::aligned_vector<GyroData> &get_gyro_data() const {
    return gyro_data;
  }
  const std::vector<int64_t> &get_gt_timestamps() const {
    return gt_timestamps;
  }
  const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data() const {
    return gt_pose_data;
  }
  const Eigen::aligned_vector<Eigen::Vector3d> &get_gt_velocities() const {
    throw NotImplementedException();
  }

  int64_t get_mocap_to_imu_offset_ns() const { return mocap_to_imu_offset_ns; }

  std::vector<ImageData> get_image_data(int64_t t_ns) {
    std::vector<ImageData> res(num_cams);

    const std::vector<std::string> folder = {"/image_0/", "/image_1/"};

    for (size_t i = 0; i < num_cams; i++) {
      std::string full_image_path = path + folder[i] + image_path[t_ns];

      if (fs::exists(full_image_path)) {
        cv::Mat img = cv::imread(full_image_path, cv::IMREAD_UNCHANGED);

        if (img.type() == CV_8UC1) {
          res[i].img.reset(
              new ManagedImage<PixelType>(img.cols, img.rows));

          const uint8_t *data_in = img.ptr();
          PixelType *data_out = res[i].img->ptr;

          convert_copy_gray_image(img.cols, img.rows, data_in, data_out);
        } else {
          std::cerr << "img.fmt.bpp " << img.type() << std::endl;
          std::abort();
        }
      }
    }

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend class KittiIO;
};

class KittiIO : public DatasetIoInterface {
 public:
  KittiIO() {}

  bool read(const std::string &path, size_t num_cams = 2) {
    if (!fs::exists(path)) {
      std::cerr << "No dataset found in " << path << std::endl;
      return false;
    }

    data.reset(new KittiVioDataset);

    data->num_cams = num_cams;
    data->path = path;

    read_image_timestamps(path + "/times.txt");

    if (fs::exists(path + "/poses.txt")) {
      read_gt_data_pose(path + "/poses.txt");
    }
    return true;
  }

  void reset() { data.reset(); }

  VioDatasetPtr get_data() { return data; }

 private:
  void read_image_timestamps(const std::string &path) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;
      std::stringstream ss(line);

      double t_s;
      ss >> t_s;

      int64_t t_ns = t_s * 1e9;

      std::stringstream ss1;
      ss1 << std::setfill('0') << std::setw(6) << data->image_timestamps.size()
          << ".png";

      data->image_timestamps.emplace_back(t_ns);
      data->image_path[t_ns] = ss1.str();
    }
  }

  void read_gt_data_pose(const std::string &path) {
    data->gt_timestamps.clear();
    data->gt_pose_data.clear();

    int i = 0;

    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      Eigen::Matrix3d rot;
      Eigen::Vector3d pos;

      ss >> rot(0, 0) >> rot(0, 1) >> rot(0, 2) >> pos[0] >> rot(1, 0) >>
          rot(1, 1) >> rot(1, 2) >> pos[1] >> rot(2, 0) >> rot(2, 1) >>
          rot(2, 2) >> pos[2];

      data->gt_timestamps.emplace_back(data->image_timestamps[i]);
      data->gt_pose_data.emplace_back(Eigen::Quaterniond(rot), pos);
      i++;
    }
  }

  std::shared_ptr<KittiVioDataset> data;
};

}  // namespace granite
