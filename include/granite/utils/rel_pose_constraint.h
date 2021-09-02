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

#include <sophus/se3.hpp>

#include <granite/imu/imu_types.h>
#include <granite/utils/common_types.h>

namespace granite {

/// @brief relative translation error for SE3 absolute parametrization
inline double relTranslationError(
    const double norm_should, const Sophus::SE3d& T_w_i,
    const Sophus::SE3d& T_w_j,
    Eigen::Matrix<double, 1, 3>* d_res_d_T_w_i = nullptr,
    Eigen::Matrix<double, 1, 3>* d_res_d_T_w_j = nullptr) {
  const Sophus::SE3d T_j_w = T_w_j.inverse();
  const Sophus::SE3d T_j_i = T_j_w * T_w_i;
  const double norm_is = T_j_i.translation().norm();
  double res = norm_is - norm_should;

  if (d_res_d_T_w_i || d_res_d_T_w_j) {
    const Eigen::Matrix<double, 1, 3> der = 1.0 / norm_is *
                                            T_j_i.translation().transpose() *
                                            T_j_w.rotationMatrix();

    if (d_res_d_T_w_i) {
      *d_res_d_T_w_i = der;
    }

    if (d_res_d_T_w_j) {
      *d_res_d_T_w_j = -der;
    }
  }

  return res;
}

/// @brief relative translation error for SE3 relative parametrization
inline double relTranslationErrorSE3(
    const double norm_should, const Eigen::aligned_vector<Sophus::SE3d>& T_i_js,
    Eigen::aligned_vector<Eigen::Matrix<double, 1, se3_SIZE>>* d_res_d_xis =
        nullptr) {
  Sophus::SE3d T_i_j;
  for (const auto& T : T_i_js) {
    T_i_j *= T;
  }
  Sophus::SE3d T_j_i = T_i_j.inverse();

  const double norm_is = T_j_i.translation().norm();
  double res = norm_is - norm_should;

  if (d_res_d_xis) {
    Eigen::Matrix<double, 1, se3_SIZE> d_diff_d_trans;
    d_diff_d_trans.setZero();
    d_diff_d_trans.topLeftCorner<1, 3>() =
        1.0 / norm_is * T_j_i.translation().transpose();

    GRANITE_ASSERT(T_i_js.size() == d_res_d_xis->size());

    Sophus::SE3d T_i_j_iter;
    for (ssize_t idx = T_i_js.size() - 1; idx >= 0; idx--) {
      Sophus::Matrix6d d_inc_d_xi;
      d_inc_d_xi.setIdentity();
      d_inc_d_xi.topLeftCorner<3, 3>() =
          -T_i_js.at(idx).rotationMatrix().transpose();
      d_inc_d_xi.block<3, 3>(3, 3) =
          -T_i_js.at(idx).rotationMatrix().transpose();

      (*d_res_d_xis)[idx] = d_diff_d_trans * T_i_j_iter.Adj() * d_inc_d_xi;

      T_i_j_iter *= T_i_js.at(idx).inverse();
    }
  }
  return res;
}

/// @brief relative translation error for Sim3 relative parametrization
inline double relTranslationErrorSim3(
    const double norm_should,
    const Eigen::aligned_vector<Sophus::Sim3d>& T_i_js,
    Eigen::aligned_vector<Eigen::Matrix<double, 1, sim3_SIZE>>* d_res_d_xis =
        nullptr) {
  Sophus::Sim3d T_i_j;
  for (const auto& T : T_i_js) {
    T_i_j *= T;
  }
  Sophus::Sim3d T_j_i = T_i_j.inverse();

  const double norm_is = T_j_i.translation().norm();
  double res = norm_is - norm_should;

  if (d_res_d_xis) {
    Eigen::Matrix<double, 1, sim3_SIZE> d_diff_d_trans;
    d_diff_d_trans.setZero();
    d_diff_d_trans.topLeftCorner<1, 3>() =
        1.0 / norm_is * T_j_i.translation().transpose();
    d_diff_d_trans(6) =
        1.0 / norm_is * T_j_i.translation().transpose() * T_j_i.translation();

    GRANITE_ASSERT(T_i_js.size() == d_res_d_xis->size());

    Sophus::Sim3d T_i_j_iter;
    for (ssize_t idx = T_i_js.size() - 1; idx >= 0; idx--) {
      Sophus::Matrix7d d_inc_d_xi;
      d_inc_d_xi.setIdentity();
      d_inc_d_xi.topLeftCorner<3, 3>() =
          -T_i_js.at(idx).rxso3().inverse().matrix();
      d_inc_d_xi.block<3, 3>(3, 3) =
          -T_i_js.at(idx).rotationMatrix().transpose();
      d_inc_d_xi(6, 6) = -1;

      (*d_res_d_xis)[idx] = d_diff_d_trans * T_i_j_iter.Adj() * d_inc_d_xi;

      T_i_j_iter *= T_i_js.at(idx).inverse();
    }
  }
  return res;
}

}  // namespace granite