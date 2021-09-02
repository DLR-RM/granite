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
@brief Implementation of Kannala-Brandt camera model
*/

#pragma once

#include <granite/utils/sophus_utils.hpp>

namespace granite {

/// @brief Kannala-Brandt camera model
///
/// \image html kb.png
///  The displacement of the projection from the optical center is proportional
///  to \f$d(\theta)\f$, which is a polynomial function of the angle between the
///  point and optical axis \f$\theta\f$.
/// This model has N=8 parameters \f$ \mathbf{i} = \left[f_x, f_y, c_x, c_y,
/// k_1, k_2, k_3, k_4 \right]^T \f$. See \ref project and \ref unproject
/// functions for more details. This model corresponds to fisheye camera model
/// in OpenCV.
template <typename Scalar = double>
class KannalaBrandtCamera4 {
 public:
  static constexpr int N = 8;  ///< Number of intrinsic parameters.

  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat2N = Eigen::Matrix<Scalar, 2, N>;

  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4N = Eigen::Matrix<Scalar, 4, N>;

  /// @brief Default constructor with zero intrinsics
  KannalaBrandtCamera4() { param.setZero(); }

  /// @brief Construct camera model with given vector of intrinsics
  ///
  /// @param[in] p vector of intrinsic parameters [fx, fy, cx, cy, k1, k2, k3,
  /// k4]
  explicit KannalaBrandtCamera4(const VecN& p) { param = p; }

  /// @brief Camera model name
  ///
  /// @return "kb4"
  static std::string getName() { return "kb4"; }

  /// @brief Cast to different scalar type
  template <class Scalar2>
  KannalaBrandtCamera4<Scalar2> cast() const {
    return KannalaBrandtCamera4<Scalar2>(param.template cast<Scalar2>());
  }

  /// @brief Project the point and optionally compute Jacobians
  ///
  /// Projection function is defined as follows:
  /// \f{align}{
  ///  \DeclareMathOperator{\atantwo}{atan2}
  ///  \pi(\mathbf{x}, \mathbf{i}) &=
  ///  \begin{bmatrix}
  ///  f_x ~ d(\theta) ~ \frac{x}{r}
  ///  \\ f_y ~ d(\theta) ~ \frac{y}{r}
  ///  \\ \end{bmatrix}
  ///  +
  ///  \begin{bmatrix}
  ///   c_x
  ///   \\ c_y
  /// \\ \end{bmatrix},
  ///  \\ r &= \sqrt{x^2 + y^2},
  ///  \\ \theta &= \atantwo(r, z),
  ///  \\ d(\theta) &= \theta + k_1 \theta^3 + k_2 \theta^5 + k_3 \theta^7 + k_4
  ///  \theta^9.
  /// \f}
  ///
  /// @param[in] p3d point to project
  /// @param[out] proj result of projection
  /// @param[out] d_proj_d_p3d if not nullptr computed Jacobian of projection
  /// with respect to p3d
  /// @param[out] d_proj_d_param point if not nullptr computed Jacobian of
  /// projection with respect to intrinsic parameters
  /// @return if projection is valid
  inline bool project(const Vec4& p3d, Vec2& proj,
                      Mat24* d_proj_d_p3d = nullptr,
                      Mat2N* d_proj_d_param = nullptr) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    const Scalar& x = p3d[0];
    const Scalar& y = p3d[1];
    const Scalar& z = p3d[2];

    const Scalar r2 = x * x + y * y;
    const Scalar r = sqrt(r2);

    if (r > Sophus::Constants<Scalar>::epsilonSqrt()) {
      const Scalar theta = atan2(r, z);
      const Scalar theta2 = theta * theta;

      Scalar r_theta = k4 * theta2;
      r_theta += k3;
      r_theta *= theta2;
      r_theta += k2;
      r_theta *= theta2;
      r_theta += k1;
      r_theta *= theta2;
      r_theta += 1;
      r_theta *= theta;

      const Scalar mx = x * r_theta / r;
      const Scalar my = y * r_theta / r;

      proj[0] = fx * mx + cx;
      proj[1] = fy * my + cy;

      if (d_proj_d_p3d) {
        const Scalar d_r_d_x = x / r;
        const Scalar d_r_d_y = y / r;

        const Scalar tmp = (z * z + r2);
        const Scalar d_theta_d_x = d_r_d_x * z / tmp;
        const Scalar d_theta_d_y = d_r_d_y * z / tmp;
        const Scalar d_theta_d_z = -r / tmp;

        Scalar d_r_theta_d_theta = Scalar(9) * k4 * theta2;
        d_r_theta_d_theta += Scalar(7) * k3;
        d_r_theta_d_theta *= theta2;
        d_r_theta_d_theta += Scalar(5) * k2;
        d_r_theta_d_theta *= theta2;
        d_r_theta_d_theta += Scalar(3) * k1;
        d_r_theta_d_theta *= theta2;
        d_r_theta_d_theta += Scalar(1);

        (*d_proj_d_p3d)(0, 0) =
            fx *
            (r_theta * r + x * r * d_r_theta_d_theta * d_theta_d_x -
             x * x * r_theta / r) /
            r2;
        (*d_proj_d_p3d)(1, 0) =
            fy * y * (d_r_theta_d_theta * d_theta_d_x * r - x * r_theta / r) /
            r2;

        (*d_proj_d_p3d)(0, 1) =
            fx * x * (d_r_theta_d_theta * d_theta_d_y * r - y * r_theta / r) /
            r2;

        (*d_proj_d_p3d)(1, 1) =
            fy *
            (r_theta * r + y * r * d_r_theta_d_theta * d_theta_d_y -
             y * y * r_theta / r) /
            r2;

        (*d_proj_d_p3d)(0, 2) = fx * x * d_r_theta_d_theta * d_theta_d_z / r;
        (*d_proj_d_p3d)(1, 2) = fy * y * d_r_theta_d_theta * d_theta_d_z / r;

        (*d_proj_d_p3d)(0, 3) = Scalar(0);
        (*d_proj_d_p3d)(1, 3) = Scalar(0);
      }

      if (d_proj_d_param) {
        (*d_proj_d_param).setZero();
        (*d_proj_d_param)(0, 0) = mx;
        (*d_proj_d_param)(0, 2) = Scalar(1);
        (*d_proj_d_param)(1, 1) = my;
        (*d_proj_d_param)(1, 3) = Scalar(1);

        (*d_proj_d_param)(0, 4) = fx * x * theta * theta2 / r;
        (*d_proj_d_param)(1, 4) = fy * y * theta * theta2 / r;

        d_proj_d_param->col(5) = d_proj_d_param->col(4) * theta2;
        d_proj_d_param->col(6) = d_proj_d_param->col(5) * theta2;
        d_proj_d_param->col(7) = d_proj_d_param->col(6) * theta2;
      }

    } else {
      // Check that the point is not cloze to zero norm
      if (z < Sophus::Constants<Scalar>::epsilonSqrt()) return false;

      proj[0] = fx * x / z + cx;
      proj[1] = fy * y / z + cy;

      if (d_proj_d_p3d) {
        d_proj_d_p3d->setZero();
        const Scalar z2 = z * z;

        (*d_proj_d_p3d)(0, 0) = fx / z;
        (*d_proj_d_p3d)(0, 2) = -fx * x / z2;

        (*d_proj_d_p3d)(1, 1) = fy / z;
        (*d_proj_d_p3d)(1, 2) = -fy * y / z2;
      }

      if (d_proj_d_param) {
        d_proj_d_param->setZero();
        (*d_proj_d_param)(0, 0) = x / z;
        (*d_proj_d_param)(0, 2) = Scalar(1);
        (*d_proj_d_param)(1, 1) = y / z;
        (*d_proj_d_param)(1, 3) = Scalar(1);
      }
    }

    return true;
  }

  /// @brief solve for theta using Newton's method.
  ///
  /// Find root of the polynomisl \f$ d(\theta) - r_{\theta} = 0 \f$ using
  /// Newton's method (https://en.wikipedia.org/wiki/Newton%27s_method). Used in
  /// \ref unproject function.
  ///
  /// @param ITER number of iterations
  /// @param[in] r_theta number of iterations
  /// @param[out] d_func_d_theta derivative of the function with respect to
  /// theta at the solution
  /// @returns theta - root of the polynomial
  template <int ITER>
  inline Scalar solve_theta(const Scalar& r_theta,
                            Scalar& d_func_d_theta) const {
    const Scalar& k1 = param[4];
    const Scalar& k2 = param[5];
    const Scalar& k3 = param[6];
    const Scalar& k4 = param[7];

    Scalar theta = r_theta;
    for (int i = ITER; i > 0; i--) {
      Scalar theta2 = theta * theta;

      Scalar func = k4 * theta2;
      func += k3;
      func *= theta2;
      func += k2;
      func *= theta2;
      func += k1;
      func *= theta2;
      func += 1;
      func *= theta;

      d_func_d_theta = 9 * k4 * theta2;
      d_func_d_theta += 7 * k3;
      d_func_d_theta *= theta2;
      d_func_d_theta += 5 * k2;
      d_func_d_theta *= theta2;
      d_func_d_theta += 3 * k1;
      d_func_d_theta *= theta2;
      d_func_d_theta += 1;

      // Iteration of Newton method
      theta += (r_theta - func) / d_func_d_theta;
    }

    return theta;
  }

  /// @brief Unproject the point and optionally compute Jacobians
  ///
  /// The unprojection function is computed as follows: \f{align}{
  ///     \pi^{-1}(\mathbf{u}, \mathbf{i}) &=
  ///  \begin{bmatrix}
  ///  \sin(\theta^{*}) ~ \frac{m_x}{r_{\theta}}
  ///  \\ \sin(\theta^{*}) ~ \frac{m_y}{r_{\theta}}
  ///  \\ \cos(\theta^{*})
  ///  \\ \end{bmatrix},
  ///  \\ m_x &= \frac{u - c_x}{f_x},
  ///  \\ m_y &= \frac{v - c_y}{f_y},
  ///  \\ r_{\theta} &= \sqrt{m_x^2 + m_y^2},
  ///  \\ \theta^{*} &= d^{-1}(r_{\theta}).
  /// \f}
  ///
  /// \f$\theta^{*}\f$ is the solution of \f$d(\theta)=r_{\theta}\f$. See \ref
  /// solve_theta for details.
  ///
  /// @param[in] proj point to unproject
  /// @param[out] p3d result of unprojection
  /// @param[out] d_p3d_d_proj if not nullptr computed Jacobian of unprojection
  /// with respect to proj
  /// @param[out] d_p3d_d_param point if not nullptr computed Jacobian of
  /// unprojection with respect to intrinsic parameters
  /// @return if unprojection is valid
  inline bool unproject(const Vec2& proj, Vec4& p3d,
                        Mat42* d_p3d_d_proj = nullptr,
                        Mat4N* d_p3d_d_param = nullptr) const {
    const Scalar& fx = param[0];
    const Scalar& fy = param[1];
    const Scalar& cx = param[2];
    const Scalar& cy = param[3];

    const Scalar mx = (proj[0] - cx) / fx;
    const Scalar my = (proj[1] - cy) / fy;

    Scalar theta = 0, sin_theta = 0, cos_theta = 1, thetad, scaling;
    Scalar d_func_d_theta = 0;

    scaling = 1.0;
    thetad = sqrt(mx * mx + my * my);

    if (thetad > Sophus::Constants<Scalar>::epsilonSqrt()) {
      theta = solve_theta<3>(thetad, d_func_d_theta);

      sin_theta = std::sin(theta);
      cos_theta = std::cos(theta);
      scaling = sin_theta / thetad;
    }

    p3d[0] = mx * scaling;
    p3d[1] = my * scaling;
    p3d[2] = cos_theta;
    p3d[3] = Scalar(0);

    if (d_p3d_d_proj || d_p3d_d_param) {
      Scalar d_thetad_d_mx = Scalar(0);
      Scalar d_thetad_d_my = Scalar(0);
      Scalar d_scaling_d_thetad = Scalar(0);
      Scalar d_cos_d_thetad = Scalar(0);

      Scalar d_scaling_d_k1 = Scalar(0);
      Scalar d_cos_d_k1 = Scalar(0);

      Scalar theta2 = Scalar(0);

      if (thetad > Sophus::Constants<Scalar>::epsilonSqrt()) {
        d_thetad_d_mx = mx / thetad;
        d_thetad_d_my = my / thetad;

        theta2 = theta * theta;

        d_scaling_d_thetad = (thetad * cos_theta / d_func_d_theta - sin_theta) /
                             (thetad * thetad);

        d_cos_d_thetad = sin_theta / d_func_d_theta;

        d_scaling_d_k1 =
            -cos_theta * theta * theta2 / (d_func_d_theta * thetad);

        d_cos_d_k1 = d_cos_d_thetad * theta * theta2;
      }

      const Scalar d_res0_d_mx =
          scaling + mx * d_scaling_d_thetad * d_thetad_d_mx;
      const Scalar d_res0_d_my = mx * d_scaling_d_thetad * d_thetad_d_my;

      const Scalar d_res1_d_mx = my * d_scaling_d_thetad * d_thetad_d_mx;
      const Scalar d_res1_d_my =
          scaling + my * d_scaling_d_thetad * d_thetad_d_my;

      const Scalar d_res2_d_mx = -d_cos_d_thetad * d_thetad_d_mx;
      const Scalar d_res2_d_my = -d_cos_d_thetad * d_thetad_d_my;

      Vec4 c0, c1;

      c0(0) = d_res0_d_mx / fx;
      c0(1) = d_res1_d_mx / fx;
      c0(2) = d_res2_d_mx / fx;
      c0(3) = Scalar(0);

      c1(0) = d_res0_d_my / fy;
      c1(1) = d_res1_d_my / fy;
      c1(2) = d_res2_d_my / fy;
      c1(3) = Scalar(0);

      if (d_p3d_d_proj) {
        d_p3d_d_proj->col(0) = c0;
        d_p3d_d_proj->col(1) = c1;
      }

      if (d_p3d_d_param) {
        d_p3d_d_param->setZero();

        d_p3d_d_param->col(2) = -c0;
        d_p3d_d_param->col(3) = -c1;

        d_p3d_d_param->col(0) = -c0 * mx;
        d_p3d_d_param->col(1) = -c1 * my;

        (*d_p3d_d_param)(0, 4) = mx * d_scaling_d_k1;
        (*d_p3d_d_param)(1, 4) = my * d_scaling_d_k1;
        (*d_p3d_d_param)(2, 4) = d_cos_d_k1;
        (*d_p3d_d_param)(3, 4) = Scalar(0);

        d_p3d_d_param->col(5) = d_p3d_d_param->col(4) * theta2;
        d_p3d_d_param->col(6) = d_p3d_d_param->col(5) * theta2;
        d_p3d_d_param->col(7) = d_p3d_d_param->col(6) * theta2;
      }
    }
    return true;
  }

  /// @brief Increment intrinsic parameters by inc
  ///
  /// @param[in] inc increment vector
  void operator+=(const VecN& inc) { param += inc; }

  /// @brief Returns a const reference to the intrinsic parameters vector
  ///
  /// The order is following: \f$ \left[f_x, f_y, c_x, c_y, k_1, k_2, k_3, k_4
  /// \right]^T \f$
  /// @return const reference to the intrinsic parameters vector
  const VecN& getParam() const { return param; }

  /// @brief Set parameters from initialization
  ///
  /// Initializes the camera model to  \f$ \left[f_x, f_y, c_x, c_y, 0, 0, 0, 0
  /// \right]^T \f$
  ///
  /// @param[in] init vector [fx, fy, cx, cy]
  inline void setFromInit(const Vec4& init) {
    param[0] = init[0];
    param[1] = init[1];
    param[2] = init[2];
    param[3] = init[3];
    param[4] = 0;
    param[5] = 0;
    param[6] = 0;
    param[7] = 0;
  }

  /// @brief Projections used for unit-tests
  static Eigen::aligned_vector<KannalaBrandtCamera4> getTestProjections() {
    Eigen::aligned_vector<KannalaBrandtCamera4> res;

    VecN vec1;
    vec1 << 379.045, 379.008, 505.512, 509.969, 0.00693023, -0.0013828,
        -0.000272596, -0.000452646;
    res.emplace_back(vec1);

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  VecN param;
};

}  // namespace granite
