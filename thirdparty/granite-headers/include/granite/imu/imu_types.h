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
https://gitlab.com/VladyslavUsenko/granite-headers.git

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

@file
@brief Type definitions for IMU preintegration
*/

#pragma once

#include <granite/utils/sophus_utils.hpp>
#include <memory>

namespace granite {

constexpr size_t se3_SIZE = 6;  ///< Dimentionality of the pose state
constexpr size_t sim3_SIZE = 7;
constexpr size_t se3_VEL_SIZE =
    9;  ///< Dimentionality of the pose-velocity state
constexpr size_t se3_VEL_BIAS_SIZE =
    15;  ///< Dimentionality of the pose-velocity-bias state

/// @brief State that consists of SE(3) pose at a certain time.
struct PoseState {
  using VecN = Eigen::Matrix<double, se3_SIZE, 1>;

  /// @brief Default constructor with Identity pose and zero timestamp.
  PoseState() { t_ns = 0; }

  /// @brief Constructor with timestamp and pose.
  ///
  /// @param t_ns timestamp of the state in nanoseconds
  /// @param T_w_i transformation from the body frame to the world frame
  PoseState(int64_t t_ns, const Sophus::SE3d& T_w_i)
      : t_ns(t_ns), T_w_i(T_w_i) {}

  /// @brief Apply increment to the pose
  ///
  /// For details see \ref incPose
  /// @param[in] inc 6x1 increment vector
  void applyInc(const VecN& inc) { incPose(inc, T_w_i); }

  /// @brief Apply increment to the pose
  ///
  /// The incremernt vector consists of translational and rotational parts \f$
  /// [\upsilon, \omega]^T \f$. Given the current pose \f$ R \in
  /// SO(3), p \in \mathbb{R}^3\f$ the updated pose is: \f{align}{ R' &=
  /// \exp(\omega) R
  /// \\ p' &= p + \upsilon
  /// \f}
  ///  The increment is consistent with \ref
  /// Se3Spline::applyInc.
  ///
  /// @param[in] inc 6x1 increment vector
  /// @param[in,out] T the pose to update
  inline static void incPose(const Sophus::Vector6d& inc, Sophus::SE3d& T) {
    T.translation() += inc.head<3>();
    T.so3() = Sophus::SO3d::exp(inc.tail<3>()) * T.so3();
  }

  inline static void incPose(const Sophus::Vector7d& inc, Sophus::Sim3d& T) {
    T.translation() += inc.head<3>();
    T.rxso3() = Sophus::RxSO3d::exp(inc.tail<4>()) * T.rxso3();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int64_t t_ns;        ///< timestamp of the state in nanoseconds
  Sophus::SE3d T_w_i;  ///< pose of the state
};

/// @brief State that consists of SE(3) pose and linear velocity at a certain
/// time.
struct PoseVelState : public PoseState {
  using VecN = Eigen::Matrix<double, se3_VEL_SIZE, 1>;

  /// @brief Default constructor with Identity pose and zero other values.
  PoseVelState() { vel_w_i.setZero(); };

  /// @brief Constructor with timestamp, pose and linear velocity.
  ///
  /// @param t_ns timestamp of the state in nanoseconds
  /// @param T_w_i transformation from the body frame to the world frame
  /// @param vel_w_i linear velocity in world coordinate frame
  PoseVelState(int64_t t_ns, const Sophus::SE3d& T_w_i,
               const Eigen::Vector3d& vel_w_i)
      : PoseState(t_ns, T_w_i), vel_w_i(vel_w_i) {}

  /// @brief Apply increment to the state.
  ///
  /// For pose see \ref incPose. For velocity simple addition.
  /// @param[in] inc 9x1 increment vector [trans, rot, vel]
  void applyInc(const VecN& inc) {
    PoseState::applyInc(inc.head<6>());
    vel_w_i += inc.tail<3>();
  }

  /// @brief Compute difference to other state.
  ///
  /// ```
  ///      PoseVelState::VecN inc;
  ///      PoseVelState p0, p1;
  ///      // Initialize p0 and inc
  ///      p1 = p0;
  ///      p1.applyInc(inc);
  ///      p0.diff(p1) == inc; // Should be true.
  /// ```
  /// @param other state to compute difference.
  VecN diff(const PoseVelState& other) const {
    VecN res;
    res.segment<3>(0) = other.T_w_i.translation() - T_w_i.translation();
    res.segment<3>(3) = (other.T_w_i.so3() * T_w_i.so3().inverse()).log();
    res.tail<3>() = other.vel_w_i - vel_w_i;
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d vel_w_i;  ///< Linear velocity of the state
};

/// @brief State that consists of SE(3) pose, linear velocity, gyroscope and
/// accelerometer biases at a certain time.
struct PoseVelBiasState : public PoseVelState {
  using Ptr = std::shared_ptr<PoseVelBiasState>;
  using VecN = Eigen::Matrix<double, se3_VEL_BIAS_SIZE, 1>;

  /// @brief Default constructor with Identity pose and zero other values.
  PoseVelBiasState() {
    bias_gyro.setZero();
    bias_accel.setZero();
  };

  /// @brief Constructor with timestamp, pose, linear velocity, gyroscope and
  /// accelerometer biases.
  ///
  /// @param t_ns timestamp of the state in nanoseconds
  /// @param T_w_i transformation from the body frame to the world frame
  /// @param vel_w_i linear velocity in world coordinate frame
  /// @param bias_gyro gyroscope bias
  /// @param bias_accel accelerometer bias
  PoseVelBiasState(int64_t t_ns, const Sophus::SE3d& T_w_i,
                   const Eigen::Vector3d& vel_w_i,
                   const Eigen::Vector3d& bias_gyro,
                   const Eigen::Vector3d& bias_accel)
      : PoseVelState(t_ns, T_w_i, vel_w_i),
        bias_gyro(bias_gyro),
        bias_accel(bias_accel) {}

  /// @brief Apply increment to the state.
  ///
  /// For pose see \ref incPose. For velocity and biases simple addition.
  /// @param[in] inc 15x1 increment vector [trans, rot, vel, bias_gyro,
  /// bias_accel]
  void applyInc(const VecN& inc) {
    PoseVelState::applyInc(inc.head<9>());
    bias_gyro += inc.segment<3>(9);
    bias_accel += inc.segment<3>(12);
  }

  /// @brief Compute difference to other state.
  ///
  /// ```
  ///      PoseVelBiasState::VecN inc;
  ///      PoseVelBiasState p0, p1;
  ///      // Initialize p0 and inc
  ///      p1 = p0;
  ///      p1.applyInc(inc);
  ///      p0.diff(p1) == inc; // Should be true.
  /// ```
  /// @param other state to compute difference.
  VecN diff(const PoseVelBiasState& other) const {
    VecN res;
    res.segment<9>(0) = PoseVelState::diff(other);
    res.segment<3>(9) = other.bias_gyro - bias_gyro;
    res.segment<3>(12) = other.bias_accel - bias_accel;
    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Eigen::Vector3d bias_gyro;   ///< Gyroscope bias
  Eigen::Vector3d bias_accel;  ///< Accelerometer bias
};

/// @brief Timestamped gyroscope and accelerometer measurements.
struct ImuData {
  using Ptr = std::shared_ptr<ImuData>;

  int64_t t_ns;           ///< timestamp in nanoseconds
  Eigen::Vector3d accel;  ///< Accelerometer measurement
  Eigen::Vector3d gyro;   ///< Gyroscope measurement

  /// @brief Default constructor with zero measurements.
  ImuData() {
    accel.setZero();
    gyro.setZero();
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace granite
