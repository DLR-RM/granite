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

#include <granite/camera/generic_camera.hpp>
#include <granite/camera/stereographic_param.hpp>

#include "gtest/gtest.h"
#include "test_utils.h"

template <typename CamT>
void test_project_jacobian() {
  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  using VecN = typename CamT::VecN;
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  using Mat24 = typename CamT::Mat24;
  using Mat2N = typename CamT::Mat2N;

  for (const CamT &cam : test_cams) {
    for (int x = -10; x <= 10; x++) {
      for (int y = -10; y <= 10; y++) {
        for (int z = -1; z <= 5; z++) {
          Vec4 p(x, y, z, 1);

          Mat24 Jp;
          Mat2N Jparam;

          Vec2 res1;

          bool success = cam.project(p, res1, &Jp, &Jparam);

          if (success) {
            test_jacobian(
                "d_r_d_p", Jp,
                [&](const Vec4 &x) {
                  Vec2 res;
                  cam.project(p + x, res);
                  return res;
                },
                Vec4::Zero());

            test_jacobian(
                "d_r_d_param", Jparam,
                [&](const VecN &x) {
                  Vec2 res;

                  CamT cam1 = cam;
                  cam1 += x;

                  cam1.project(p, res);
                  return res;
                },
                VecN::Zero());
          }
        }
      }
    }
  }
}

template <typename CamT>
void test_project_unproject() {
  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  using Scalar = typename CamT::Vec2::Scalar;
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  for (const CamT &cam : test_cams) {
    for (int x = -10; x <= 10; x++) {
      for (int y = -10; y <= 10; y++) {
        for (int z = 0; z <= 5; z++) {
          Vec4 p(x, y, z, 0.23424);

          Vec4 p_normalized = Vec4::Zero();
          p_normalized.template head<3>() = p.template head<3>().normalized();
          Vec2 res;
          bool success = cam.project(p, res);

          if (success) {
            Vec4 p_uproj;
            cam.unproject(res, p_uproj);

            EXPECT_TRUE(p_normalized.isApprox(
                p_uproj, Sophus::Constants<Scalar>::epsilonSqrt()))
                << "p_normalized " << p_normalized.transpose() << " p_uproj "
                << p_uproj.transpose();
          }
        }
      }
    }
  }
}

template <typename CamT>
void test_unproject_jacobians() {
  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  using VecN = typename CamT::VecN;
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  using Mat42 = typename CamT::Mat42;
  using Mat4N = typename CamT::Mat4N;

  for (const CamT &cam : test_cams) {
    for (int x = -10; x <= 10; x++) {
      for (int y = -10; y <= 10; y++) {
        for (int z = 0; z <= 5; z++) {
          Vec4 p_3d(x, y, z, 0);

          Vec2 p;
          bool success = cam.project(p_3d, p);

          if (success) {
            Mat42 Jp;
            Mat4N Jparam;

            Vec4 res1;
            cam.unproject(p, res1, &Jp, &Jparam);

            test_jacobian(
                "d_r_d_p", Jp,
                [&](const Vec2 &x) {
                  Vec4 res = Vec4::Zero();
                  cam.unproject(p + x, res);
                  return res;
                },
                Vec2::Zero());

            test_jacobian(
                "d_r_d_param", Jparam,
                [&](const VecN &x) {
                  Vec4 res = Vec4::Zero();
                  CamT cam1 = cam;
                  cam1 += x;

                  cam1.unproject(p, res);
                  return res;
                },
                VecN::Zero());
          }
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////

TEST(CameraTestCase, PinholeProjectJacobians) {
  test_project_jacobian<granite::PinholeCamera<double>>();
}
TEST(CameraTestCase, PinholeProjectJacobiansFloat) {
  test_project_jacobian<granite::PinholeCamera<float>>();
}

TEST(CameraTestCase, UnifiedProjectJacobians) {
  test_project_jacobian<granite::UnifiedCamera<double>>();
}
TEST(CameraTestCase, UnifiedProjectJacobiansFloat) {
  test_project_jacobian<granite::UnifiedCamera<float>>();
}

TEST(CameraTestCase, ExtendedUnifiedProjectJacobians) {
  test_project_jacobian<granite::ExtendedUnifiedCamera<double>>();
}
TEST(CameraTestCase, ExtendedUnifiedProjectJacobiansFloat) {
  test_project_jacobian<granite::ExtendedUnifiedCamera<float>>();
}

TEST(CameraTestCase, KannalaBrandtProjectJacobians) {
  test_project_jacobian<granite::KannalaBrandtCamera4<double>>();
}
TEST(CameraTestCase, KannalaBrandtProjectJacobiansFloat) {
  test_project_jacobian<granite::KannalaBrandtCamera4<float>>();
}

TEST(CameraTestCase, DoubleSphereJacobians) {
  test_project_jacobian<granite::DoubleSphereCamera<double>>();
}
TEST(CameraTestCase, FovCameraJacobians) {
  test_project_jacobian<granite::FovCamera<double>>();
}

TEST(CameraTestCase, BalCameraJacobians) {
  test_project_jacobian<granite::BalCamera<double>>();
}

TEST(CameraTestCase, BalCameraJacobiansFloat) {
  test_project_jacobian<granite::BalCamera<float>>();
}

////////////////////////////////////////////////////////////////

TEST(CameraTestCase, PinholeProjectUnproject) {
  test_project_unproject<granite::PinholeCamera<double>>();
}
TEST(CameraTestCase, PinholeProjectUnprojectFloat) {
  test_project_unproject<granite::PinholeCamera<float>>();
}

TEST(CameraTestCase, UnifiedProjectUnproject) {
  test_project_unproject<granite::UnifiedCamera<double>>();
}
TEST(CameraTestCase, UnifiedProjectUnprojectFloat) {
  test_project_unproject<granite::UnifiedCamera<float>>();
}

TEST(CameraTestCase, ExtendedUnifiedProjectUnproject) {
  test_project_unproject<granite::ExtendedUnifiedCamera<double>>();
}
TEST(CameraTestCase, ExtendedUnifiedProjectUnprojectFloat) {
  test_project_unproject<granite::ExtendedUnifiedCamera<float>>();
}

TEST(CameraTestCase, KannalaBrandtProjectUnproject) {
  test_project_unproject<granite::KannalaBrandtCamera4<double>>();
}
TEST(CameraTestCase, KannalaBrandtProjectUnprojectFloat) {
  test_project_unproject<granite::KannalaBrandtCamera4<float>>();
}

TEST(CameraTestCase, DoubleSphereProjectUnproject) {
  test_project_unproject<granite::DoubleSphereCamera<double>>();
}
TEST(CameraTestCase, DoubleSphereProjectUnprojectFloat) {
  test_project_unproject<granite::DoubleSphereCamera<float>>();
}

TEST(CameraTestCase, FovProjectUnproject) {
  test_project_unproject<granite::FovCamera<double>>();
}

TEST(CameraTestCase, FovProjectUnprojectFloat) {
  test_project_unproject<granite::FovCamera<float>>();
}

TEST(CameraTestCase, BalProjectUnproject) {
  test_project_unproject<granite::BalCamera<double>>();
}

TEST(CameraTestCase, BalProjectUnprojectFloat) {
  test_project_unproject<granite::BalCamera<float>>();
}

/////////////////////////////////////////////////////////////////////////

TEST(CameraTestCase, PinholeUnprojectJacobians) {
  test_unproject_jacobians<granite::PinholeCamera<double>>();
}
TEST(CameraTestCase, PinholeUnprojectJacobiansFloat) {
  test_unproject_jacobians<granite::PinholeCamera<float>>();
}

TEST(CameraTestCase, UnifiedUnprojectJacobians) {
  test_unproject_jacobians<granite::UnifiedCamera<double>>();
}
TEST(CameraTestCase, UnifiedUnprojectJacobiansFloat) {
  test_unproject_jacobians<granite::UnifiedCamera<float>>();
}

TEST(CameraTestCase, ExtendedUnifiedUnprojectJacobians) {
  test_unproject_jacobians<granite::ExtendedUnifiedCamera<double>>();
}
TEST(CameraTestCase, ExtendedUnifiedUnprojectJacobiansFloat) {
  test_unproject_jacobians<granite::ExtendedUnifiedCamera<float>>();
}

TEST(CameraTestCase, KannalaBrandtUnprojectJacobians) {
  test_unproject_jacobians<granite::KannalaBrandtCamera4<double>>();
}
// TEST(CameraTestCase, KannalaBrandtUnprojectJacobiansFloat) {
//  test_unproject_jacobians<granite::KannalaBrandtCamera4<float>>();
//}

TEST(CameraTestCase, DoubleSphereUnprojectJacobians) {
  test_unproject_jacobians<granite::DoubleSphereCamera<double>>();
}
// TEST(CameraTestCase, DoubleSphereUnprojectJacobiansFloat) {
//  test_unproject_jacobians<granite::DoubleSphereCamera<float>>();
//}

TEST(CameraTestCase, FovUnprojectJacobians) {
  test_unproject_jacobians<granite::FovCamera<double>>();
}
TEST(CameraTestCase, FovUnprojectJacobiansFloat) {
  test_unproject_jacobians<granite::FovCamera<float>>();
}

////////////////////////////////////////////////////////////////

template <typename CamT>
void test_stereographic_project_jacobian() {
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;
  using Mat24 = typename CamT::Mat24;

  for (int x = -10; x <= 10; x++) {
    for (int y = -10; y <= 10; y++) {
      Vec4 p(x, y, 5, 0.23424);

      Mat24 Jp;

      Vec2 res1 = CamT::project(p, &Jp);
      Vec2 res2 = CamT::project(p);

      ASSERT_TRUE(res1.isApprox(res2))
          << "res1 " << res1.transpose() << " res2 " << res2.transpose();

      test_jacobian(
          "d_r_d_p", Jp, [&](const Vec4 &x) { return CamT::project(p + x); },
          Vec4::Zero());
    }
  }
}

template <typename CamT>
void test_stereographic_project_unproject() {
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  for (int x = -10; x <= 10; x++) {
    for (int y = -10; y <= 10; y++) {
      Vec4 p(x, y, 5, 0.23424);

      Vec4 p_normalized = Vec4::Zero();
      p_normalized.template head<3>() = p.template head<3>().normalized();
      Vec2 res = CamT::project(p);
      Vec4 p_uproj = CamT::unproject(res);

      ASSERT_TRUE(p_normalized.isApprox(p_uproj))
          << "p_normalized " << p_normalized.transpose() << " p_uproj "
          << p_uproj.transpose();
    }
  }
}

template <typename CamT>
void test_stereographic_unproject_jacobian() {
  using Vec2 = typename CamT::Vec2;
  using Vec4 = typename CamT::Vec4;

  using Mat42 = typename CamT::Mat42;

  for (int x = -10; x <= 10; x++) {
    for (int y = -10; y <= 10; y++) {
      Vec4 p_3d(x, y, 5, 0.23424);

      Vec2 p = CamT::project(p_3d);

      Mat42 Jp;

      Vec4 res1 = CamT::unproject(p, &Jp);
      Vec4 res2 = CamT::unproject(p);

      ASSERT_TRUE(res1.isApprox(res2))
          << "res1 " << res1.transpose() << " res2 " << res2.transpose();

      test_jacobian(
          "d_r_d_p", Jp, [&](const Vec2 &x) { return CamT::unproject(p + x); },
          Vec2::Zero());
    }
  }
}

TEST(CameraTestCase, StereographicParamProjectJacobians) {
  test_stereographic_project_jacobian<granite::StereographicParam<double>>();
}
TEST(CameraTestCase, StereographicParamProjectJacobiansFloat) {
  test_stereographic_project_jacobian<granite::StereographicParam<float>>();
}

TEST(CameraTestCase, StereographicParamProjectUnproject) {
  test_stereographic_project_unproject<granite::StereographicParam<double>>();
}
TEST(CameraTestCase, StereographicParamProjectUnprojectFloat) {
  test_stereographic_project_unproject<granite::StereographicParam<float>>();
}

TEST(CameraTestCase, StereographicParamUnprojectJacobians) {
  test_stereographic_unproject_jacobian<granite::StereographicParam<double>>();
}
TEST(CameraTestCase, StereographicParamUnprojectJacobiansFloat) {
  test_stereographic_unproject_jacobian<granite::StereographicParam<float>>();
}
