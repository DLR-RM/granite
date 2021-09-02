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

#include <granite/utils/sophus_utils.hpp>

#include <pangolin/gl/gldraw.h>
#include <Eigen/Dense>
#include <sophus/interpolate.hpp>

const u_int8_t cam_color[3]{250, 0, 26};
const u_int8_t state_color[3]{250, 0, 26};
const u_int8_t pose_color[3]{0, 50, 255};
const u_int8_t gt_color[3]{0, 171, 47};

inline void render_camera(const Eigen::Matrix4d& T_w_c, float lineWidth,
                          const u_int8_t* color, float sizeFactor) {
  const float sz = sizeFactor;
  const float width = 640, height = 480, fx = 500, fy = 500, cx = 320, cy = 240;

  const Eigen::aligned_vector<Eigen::Vector3f> lines = {
      {0, 0, 0},
      {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
      {0, 0, 0},
      {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {0, 0, 0},
      {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {0, 0, 0},
      {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz}};

  glPushMatrix();
  glMultMatrixd(T_w_c.data());
  glColor3ubv(color);
  glLineWidth(lineWidth);
  pangolin::glDrawLines(lines);
  glPopMatrix();
}

inline void getcolor(float p, float np, float& r, float& g, float& b) {
  float inc = 4.0 / np;
  float x = p * inc;
  r = 0.0f;
  g = 0.0f;
  b = 0.0f;

  if ((0 <= x && x <= 1) || (5 <= x && x <= 6))
    r = 1.0f;
  else if (4 <= x && x <= 5)
    r = x - 4;
  else if (1 <= x && x <= 2)
    r = 1.0f - (x - 1);

  if (1 <= x && x <= 3)
    g = 1.0f;
  else if (0 <= x && x <= 1)
    g = x - 0;
  else if (3 <= x && x <= 4)
    g = 1.0f - (x - 3);

  if (3 <= x && x <= 5)
    b = 1.0f;
  else if (2 <= x && x <= 3)
    b = x - 2;
  else if (5 <= x && x <= 6)
    b = 1.0f - (x - 5);
}

inline Sophus::SE3d interpolate_trajectory(
    const std::vector<int64_t>& t_ns,
    const Eigen::aligned_vector<Sophus::SE3d>& trajectory,
    const int64_t timestamp) {
  // do we even have to interpolate?
  if (timestamp <= t_ns.front()) {
    return *trajectory.begin();
  }

  if (timestamp >= t_ns.back()) {
    return *trajectory.rbegin();
  }

  const auto exact_time_it = std::find(t_ns.begin(), t_ns.end(), timestamp);
  if (exact_time_it != t_ns.end()) {
    return trajectory.at(std::distance(t_ns.begin(), exact_time_it));
  }

  // interpolate
  // find closest element
  const auto closest_time_min_it = std::min_element(
      t_ns.begin(), t_ns.end(), [&](const auto& ts1, const auto& ts2) {
        return std::abs(ts1 - timestamp) < std::abs(ts2 - timestamp);
      });

  const auto closest_smaller_time_it = (*closest_time_min_it < timestamp)
                                           ? closest_time_min_it
                                           : std::prev(closest_time_min_it);
  const auto closest_bigger_time_it = (*closest_time_min_it < timestamp)
                                          ? std::next(closest_time_min_it)
                                          : closest_time_min_it;

  const double interpolation_factor =
      (double(timestamp * 1e-6) - double(*closest_smaller_time_it * 1e-6)) /
      (double(*closest_bigger_time_it * 1e-6) -
       double(*closest_smaller_time_it * 1e-6));

  return Sophus::interpolate(
      trajectory.at(std::distance(t_ns.begin(), closest_smaller_time_it)),
      trajectory.at(std::distance(t_ns.begin(), closest_bigger_time_it)),
      interpolation_factor);
}
