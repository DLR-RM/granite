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
@brief Implementation of generic camera model
*/

#pragma once

#include <granite/camera/bal_camera.hpp>
#include <granite/camera/double_sphere_camera.hpp>
#include <granite/camera/extended_camera.hpp>
#include <granite/camera/fov_camera.hpp>
#include <granite/camera/kannala_brandt_camera4.hpp>
#include <granite/camera/pinhole_camera.hpp>
#include <granite/camera/unified_camera.hpp>

#include <variant>

namespace granite {

/// @brief Generic camera model that can store different camera models
///
/// Particular class of camera model is stored as \ref variant and can be casted
/// to specific type using std::visit.
template <typename Scalar>
class GenericCamera {
  using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using Vec4 = Eigen::Matrix<Scalar, 4, 1>;
  using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

  using Mat24 = Eigen::Matrix<Scalar, 2, 4>;
  using Mat42 = Eigen::Matrix<Scalar, 4, 2>;
  using Mat4 = Eigen::Matrix<Scalar, 4, 4>;

  using VariantT =
      std::variant<ExtendedUnifiedCamera<Scalar>, DoubleSphereCamera<Scalar>,
                   KannalaBrandtCamera4<Scalar>, UnifiedCamera<Scalar>,
                   PinholeCamera<Scalar>>;  ///< Possible variants of camera
                                            ///< models.

 public:
  /// @brief Cast to different scalar type
  template <typename Scalar2>
  inline GenericCamera<Scalar2> cast() const {
    GenericCamera<Scalar2> res;
    std::visit([&](const auto& v) { res.variant = v.template cast<Scalar2>(); },
               variant);
    return res;
  }

  /// @brief Number of intrinsic parameters
  inline int getN() const {
    int res;
    std::visit([&](const auto& v) { res = v.N; }, variant);
    return res;
  }

  /// @brief Camera model name
  inline std::string getName() const {
    std::string res;
    std::visit([&](const auto& v) { res = v.getName(); }, variant);
    return res;
  }

  /// @brief Set parameters from initialization
  ///
  /// @param[in] init vector [fx, fy, cx, cy]
  inline void setFromInit(const Vec4& init) {
    std::visit([&](auto& v) { return v.setFromInit(init); }, variant);
  }

  /// @brief Increment intrinsic parameters by inc and if necessary clamp the
  /// values to the valid range
  inline void applyInc(const VecX& inc) {
    std::visit([&](auto& v) { return v += inc; }, variant);
  }

  /// @brief Returns a vector of intrinsic parameters
  ///
  /// The order of parameters depends on the stored model.
  /// @return vector of intrinsic parameters vector
  inline VecX getParam() const {
    VecX res;
    std::visit([&](const auto& cam) { res = cam.getParam(); }, variant);
    return res;
  }

  /// @brief Project a single point and optionally compute Jacobian
  ///
  /// **SLOW** function, as it requires vtable lookup for every projection.
  ///
  /// @param[in] p3d point to project
  /// @param[out] proj result of projection
  /// @param[out] d_proj_d_p3d if not nullptr computed Jacobian of projection
  /// with respect to p3d
  /// @return if projection is valid
  inline bool project(const Vec4& p3d, Vec2& proj,
                      Mat24* d_proj_d_p3d = nullptr) const {
    bool res;
    std::visit(
        [&](const auto& cam) { res = cam.project(p3d, proj, d_proj_d_p3d); },
        variant);
    return res;
  }

  /// @brief Unproject a single point and optionally compute Jacobian
  ///
  /// **SLOW** function, as it requires vtable lookup for every unprojection.
  ///
  /// @param[in] proj point to unproject
  /// @param[out] p3d result of unprojection
  /// @param[out] d_p3d_d_proj if not nullptr computed Jacobian of unprojection
  /// with respect to proj
  /// @return if unprojection is valid
  inline bool unproject(const Vec2& proj, Vec4& p3d,
                        Mat42* d_p3d_d_proj = nullptr) const {
    bool res;
    std::visit(
        [&](const auto& cam) { res = cam.unproject(proj, p3d, d_p3d_d_proj); },
        variant);
    return res;
  }

  /// @brief Project a vector of points
  ///
  /// @param[in] p3d points to project
  /// @param[in] T_c_w transformation from world to camera frame that should be
  /// applied to points before projection
  /// @param[out] proj results of projection
  /// @param[out] proj_success if projection is valid
  inline void project(const Eigen::aligned_vector<Vec3>& p3d, const Mat4& T_c_w,
                      Eigen::aligned_vector<Vec2>& proj,
                      std::vector<bool>& proj_success) const {
    std::visit(
        [&](const auto& cam) {
          proj.resize(p3d.size());
          proj_success.resize(p3d.size());
          for (size_t i = 0; i < p3d.size(); i++) {
            proj_success[i] =
                cam.project(T_c_w * p3d[i].homogeneous(), proj[i]);
          }
        },
        variant);
  }

  /// @brief Project a vector of points
  ///
  /// @param[in] p3d points to project
  /// @param[in] T_c_w transformation from world to camera frame that should be
  /// applied to points before projection
  /// @param[out] proj results of projection
  /// @param[out] proj_success if projection is valid
  inline void project(const Eigen::aligned_vector<Vec4>& p3d, const Mat4& T_c_w,
                      Eigen::aligned_vector<Vec2>& proj,
                      std::vector<bool>& proj_success) const {
    std::visit(
        [&](const auto& cam) {
          proj.resize(p3d.size());
          proj_success.resize(p3d.size());
          for (size_t i = 0; i < p3d.size(); i++) {
            proj_success[i] = cam.project(T_c_w * p3d[i], proj[i]);
          }
        },
        variant);
  }

  /// @brief Project a vector of points
  ///
  /// @param[in] p3d points to project
  /// @param[out] proj results of projection
  /// @param[out] proj_success if projection is valid
  inline void project(const Eigen::aligned_vector<Vec4>& p3d,
                      Eigen::aligned_vector<Vec2>& proj,
                      std::vector<bool>& proj_success) const {
    std::visit(
        [&](const auto& cam) {
          proj.resize(p3d.size());
          proj_success.resize(p3d.size());
          for (size_t i = 0; i < p3d.size(); i++) {
            proj_success[i] = cam.project(p3d[i], proj[i]);
          }
        },
        variant);
  }

  /// @brief Project a vector of points, compute polar and azimuthal angles
  ///
  /// @param[in] p3d points to project
  /// @param[in] T_c_w transformation from world to camera frame that should be
  /// applied to points before projection
  /// @param[out] proj results of projection
  /// @param[out] proj_success if projection is valid
  inline void project(
      const Eigen::aligned_vector<Vec4>& p3d, const Mat4& T_c_w,
      Eigen::aligned_vector<Vec2>& proj, std::vector<bool>& proj_success,
      Eigen::aligned_vector<Vec2>& polar_azimuthal_angle) const {
    std::visit(
        [&](const auto& cam) {
          proj.resize(p3d.size());
          proj_success.resize(p3d.size());
          polar_azimuthal_angle.resize(p3d.size());
          for (size_t i = 0; i < p3d.size(); i++) {
            Vec4 p3dt = T_c_w * p3d[i];

            proj_success[i] = cam.project(p3dt, proj[i]);

            Scalar r2 = p3dt[0] * p3dt[0] + p3dt[1] * p3dt[1];
            Scalar r = std::sqrt(r2);
            polar_azimuthal_angle[i][0] = std::atan2(r, p3dt[2]);
            polar_azimuthal_angle[i][1] = std::atan2(p3dt[0], p3dt[1]);
          }
        },
        variant);
  }

  /// @brief Unproject a vector of points
  ///
  /// @param[in] proj points to unproject
  /// @param[out] p3d results of unprojection
  /// @param[out] unproj_success if unprojection is valid
  inline void unproject(const Eigen::aligned_vector<Vec2>& proj,
                        Eigen::aligned_vector<Vec4>& p3d,
                        std::vector<bool>& unproj_success) const {
    std::visit(
        [&](const auto& cam) {
          p3d.resize(proj.size());
          unproj_success.resize(proj.size());
          for (size_t i = 0; i < p3d.size(); i++) {
            unproj_success[i] = cam.unproject(proj[i], p3d[i]);
          }
        },
        variant);
  }

  /// @brief Construct a particular type of camera model from name
  static GenericCamera<Scalar> fromString(const std::string& name) {
    GenericCamera<Scalar> res;

    constexpr size_t variant_size = std::variant_size<VariantT>::value;
    visitAllTypes<variant_size - 1>(res, name);

    return res;
  }

  VariantT variant;

 private:
  /// @brief Iterate over all possible types of the variant and construct that
  /// type that has a matching name
  template <int I>
  static void visitAllTypes(GenericCamera<Scalar>& res,
                            const std::string& name) {
    if constexpr (I >= 0) {
      using cam_t = typename std::variant_alternative<I, VariantT>::type;
      if (cam_t::getName() == name) {
        cam_t val;
        res.variant = val;
      }
      visitAllTypes<I - 1>(res, name);
    }
  }
};
}  // namespace granite
