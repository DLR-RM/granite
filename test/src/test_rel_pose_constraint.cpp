#include <granite/utils/rel_pose_constraint.h>

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(PreIntegrationTestSuite, RelTranslationTest) {
  Sophus::SE3d T_w_i = Sophus::se3_expd(Sophus::Vector6d::Random());
  Sophus::SE3d T_w_j = Sophus::se3_expd(Sophus::Vector6d::Random());

  Sophus::SE3d T_i_j = Sophus::se3_expd(Sophus::Vector6d::Random() / 100) *
                       T_w_i.inverse() * T_w_j;

  Eigen::Matrix<double, 1, 3> d_res_d_T_w_i, d_res_d_T_w_j;
  granite::relTranslationError(T_i_j.translation().norm(), T_w_i, T_w_j,
                              &d_res_d_T_w_i, &d_res_d_T_w_j);

  {
    Eigen::Vector3d x0;
    x0.setZero();
    test_gradient(
        "d_res_d_T_w_i", d_res_d_T_w_i,
        [&](const Eigen::Vector3d& x) {
          auto T_w_i_new = T_w_i;
          Sophus::Vector6d xi;
          xi.head<3>() = x;
          xi.tail<3>().setZero();
          granite::PoseState::incPose(xi, T_w_i_new);

          return granite::relTranslationError(T_i_j.translation().norm(),
                                             T_w_i_new, T_w_j);
        },
        x0);
  }

  {
    Eigen::Vector3d x0;
    x0.setZero();
    test_gradient(
        "d_res_d_T_w_j", d_res_d_T_w_j,
        [&](const Eigen::Vector3d& x) {
          auto T_w_j_new = T_w_j;
          Sophus::Vector6d xi;
          xi.head<3>() = x;
          xi.tail<3>().setZero();
          granite::PoseState::incPose(xi, T_w_j_new);

          return granite::relTranslationError(T_i_j.translation().norm(), T_w_i,
                                             T_w_j_new);
        },
        x0);
  }
}

TEST(PreIntegrationTestSuite, RelTranslationTestSE3) {
  const Sophus::SE3d T_a_b = Sophus::se3_expd(Sophus::Vector6d::Random());
  const Sophus::SE3d T_b_c = Sophus::se3_expd(Sophus::Vector6d::Random());
  const Sophus::SE3d T_c_d = Sophus::se3_expd(Sophus::Vector6d::Random());

  const Eigen::aligned_vector<Sophus::SE3d> T_a_ds = {T_a_b, T_b_c, T_c_d};
  Eigen::aligned_vector<Eigen::Matrix<double, 1, granite::se3_SIZE>> d_res_d_xis(
      T_a_ds.size());

  granite::relTranslationErrorSE3(5.0, T_a_ds, &d_res_d_xis);

  ASSERT_TRUE(
      (T_a_b * T_b_c * T_c_d)
          .inverse()
          .translation()
          .isApprox((T_c_d.inverse() * T_b_c.inverse() * T_a_b.inverse())
                        .translation()));

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_gradient(
        "d_res_d_xis 0", d_res_d_xis.at(0),
        [&](const Sophus::Vector6d& xi) {
          auto T_a_b_new = T_a_b;
          granite::PoseState::incPose(xi, T_a_b_new);
          const Eigen::aligned_vector<Sophus::SE3d> T_a_ds_new = {T_a_b_new,
                                                                  T_b_c, T_c_d};

          return granite::relTranslationErrorSE3(5.0, T_a_ds_new);
        },
        x0);
  }

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_gradient(
        "d_res_d_xis 1", d_res_d_xis.at(1),
        [&](const Sophus::Vector6d& xi) {
          auto T_b_c_new = T_b_c;
          granite::PoseState::incPose(xi, T_b_c_new);
          const Eigen::aligned_vector<Sophus::SE3d> T_a_ds_new = {
              T_a_b, T_b_c_new, T_c_d};

          return granite::relTranslationErrorSE3(5.0, T_a_ds_new);
        },
        x0);
  }

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_gradient(
        "d_res_d_xis 2", d_res_d_xis.at(2),
        [&](const Sophus::Vector6d& xi) {
          auto T_c_d_new = T_c_d;
          granite::PoseState::incPose(xi, T_c_d_new);
          const Eigen::aligned_vector<Sophus::SE3d> T_a_ds_new = {T_a_b, T_b_c,
                                                                  T_c_d_new};

          return granite::relTranslationErrorSE3(5.0, T_a_ds_new);
        },
        x0);
  }
}

TEST(PreIntegrationTestSuite, RelTranslationTestSim3) {
  const Sophus::Sim3d T_a_b = Sophus::sim3_expd(Sophus::Vector7d::Random());
  const Sophus::Sim3d T_b_c = Sophus::sim3_expd(Sophus::Vector7d::Random());
  const Sophus::Sim3d T_c_d = Sophus::sim3_expd(Sophus::Vector7d::Random());

  const Eigen::aligned_vector<Sophus::Sim3d> T_a_ds = {T_a_b, T_b_c, T_c_d};
  Eigen::aligned_vector<Eigen::Matrix<double, 1, granite::sim3_SIZE>>
      d_res_d_xis(T_a_ds.size());

  granite::relTranslationErrorSim3(5.0, T_a_ds, &d_res_d_xis);

  ASSERT_TRUE(
      (T_a_b * T_b_c * T_c_d)
          .inverse()
          .translation()
          .isApprox((T_c_d.inverse() * T_b_c.inverse() * T_a_b.inverse())
                        .translation()));

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_gradient(
        "d_res_d_xis 0", d_res_d_xis.at(0),
        [&](const Sophus::Vector7d& xi) {
          auto T_a_b_new = T_a_b;
          granite::PoseState::incPose(xi, T_a_b_new);
          const Eigen::aligned_vector<Sophus::Sim3d> T_a_ds_new = {
              T_a_b_new, T_b_c, T_c_d};

          return granite::relTranslationErrorSim3(5.0, T_a_ds_new);
        },
        x0);
  }

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_gradient(
        "d_res_d_xis 1", d_res_d_xis.at(1),
        [&](const Sophus::Vector7d& xi) {
          auto T_b_c_new = T_b_c;
          granite::PoseState::incPose(xi, T_b_c_new);
          const Eigen::aligned_vector<Sophus::Sim3d> T_a_ds_new = {
              T_a_b, T_b_c_new, T_c_d};

          return granite::relTranslationErrorSim3(5.0, T_a_ds_new);
        },
        x0);
  }

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_gradient(
        "d_res_d_xis 2", d_res_d_xis.at(2),
        [&](const Sophus::Vector7d& xi) {
          auto T_c_d_new = T_c_d;
          granite::PoseState::incPose(xi, T_c_d_new);
          const Eigen::aligned_vector<Sophus::Sim3d> T_a_ds_new = {T_a_b, T_b_c,
                                                                   T_c_d_new};

          return granite::relTranslationErrorSim3(5.0, T_a_ds_new);
        },
        x0);
  }
}