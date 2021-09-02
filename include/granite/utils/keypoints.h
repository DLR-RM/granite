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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <granite/image/image.h>
#include <granite/utils/sophus_utils.hpp>

#include <granite/utils/common_types.h>
#include <granite/camera/generic_camera.hpp>

namespace granite {

typedef std::bitset<256> Descriptor;

void detectKeypointsMapping(const granite::Image<const PixelType>& img_raw,
                            KeypointsData& kd, int num_features);

void detectKeypoints(
    const granite::Image<const PixelType>& img_raw, KeypointsData& kd,
    int PATCH_SIZE = 32, int num_points_cell = 1,
    const Eigen::aligned_vector<Eigen::Vector2d>& current_points =
        Eigen::aligned_vector<Eigen::Vector2d>());

void computeAngles(const granite::Image<const PixelType>& img_raw,
                   KeypointsData& kd, bool rotate_features);

void computeDescriptors(const granite::Image<const PixelType>& img_raw,
                        KeypointsData& kd);

void matchDescriptors(const std::vector<std::bitset<256>>& corner_descriptors_1,
                      const std::vector<std::bitset<256>>& corner_descriptors_2,
                      std::vector<std::pair<int, int>>& matches, int threshold,
                      double dist_2_best);

inline void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix4d& E) {
  E.setZero();
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  E.topLeftCorner<3, 3>() = Sophus::SO3d::hat(t_0_1.normalized()) * R_0_1;
}

inline void findInliersEssential(const KeypointsData& kd1,
                                 const KeypointsData& kd2,
                                 const Eigen::Matrix4d& E,
                                 double epipolar_error_threshold,
                                 MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector4d p0_3d = kd1.corners_3d[md.matches[j].first];
    const Eigen::Vector4d p1_3d = kd2.corners_3d[md.matches[j].second];

    const double epipolar_error = std::abs(p0_3d.transpose() * E * p1_3d);

    if (epipolar_error < epipolar_error_threshold) {
      md.inliers.push_back(md.matches[j]);
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md);

}  // namespace granite
