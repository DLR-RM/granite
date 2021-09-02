#include <granite/vi_estimator/mono_map_initialization.h>

#include "gtest/gtest.h"
#include "test_utils.h"

TEST(VioTestSuite, HomographyTest) {
  Eigen::aligned_vector<Eigen::Vector3d> points_3d;
  // for (int x = -3; x <= 3; x++) {
  //   for (int y = -3; y <= 3; y++) {
  //     points_3d.push_back(Eigen::Vector3d(x, y, 0.0));
  //   }
  // }
  points_3d.push_back(Eigen::Vector3d(5.2, 3.5, 0.0));
  points_3d.push_back(Eigen::Vector3d(-3.7, 1.1, 0.0));
  points_3d.push_back(Eigen::Vector3d(2.9, 0.8, 0.0));
  points_3d.push_back(Eigen::Vector3d(-3.5, -5.2, 0.0));

  const Sophus::SE3d T_w_cama =
      Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(2.0, 4.0, 1.0));

  const Sophus::SE3d T_w_camb =
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d(0.12, 0.31, -0.41)),
                   Eigen::Vector3d(2.0, 5.0, -1.0));

  Eigen::Matrix3d H_cama_w;
  H_cama_w.col(0) = T_w_cama.inverse().rotationMatrix().col(0);
  H_cama_w.col(1) = T_w_cama.inverse().rotationMatrix().col(1);
  H_cama_w.col(2) = T_w_cama.inverse().translation();

  Eigen::Matrix3d H_camb_w;
  H_camb_w.col(0) = T_w_camb.inverse().rotationMatrix().col(0);
  H_camb_w.col(1) = T_w_camb.inverse().rotationMatrix().col(1);
  H_camb_w.col(2) = T_w_camb.inverse().translation();

  const Sophus::SE3d T_cama_camb = T_w_cama.inverse() * T_w_camb;

  Eigen::Matrix3d H = H_cama_w * H_camb_w.inverse();

  H /= H(2, 2);

  Eigen::aligned_vector<Eigen::Vector3d> bearings_a, bearings_b;
  for (const auto& point_3d : points_3d) {
    const Eigen::Vector3d bearing_vector_a =
        (T_w_cama.inverse() * point_3d).normalized();
    bearings_a.push_back(bearing_vector_a);
    const Eigen::Vector3d bearing_vector_b =
        (T_w_camb.inverse() * point_3d).normalized();
    bearings_b.push_back(bearing_vector_b);
  }

  Eigen::Vector3d b_check = (H * bearings_b.at(0)).normalized();
  if (b_check.transpose() * bearings_a.at(0) < 0) {
    b_check *= -1;
  }
  ASSERT_TRUE(bearings_a.at(0).isApprox(b_check)) << bearings_a.at(0) << "\n\n"
                                                  << b_check;

  // std::vector<int> inlier_idxs;
  // std::optional<Sophus::SE3d> T_cama_camb_est =
  //     granite::relative_pose_homography(bearings_a, bearings_b, 0.01,
  //                                      &inlier_idxs);

  std::optional<Eigen::Matrix3d> H_est_opt =
      granite::compute_homography(bearings_a, bearings_b);
  ASSERT_TRUE(H_est_opt.has_value()) << "Homography estimation failed";

  Eigen::Matrix3d H_est = H_est_opt.value();

  H_est /= H_est(2, 2);

  ASSERT_TRUE(H.isApprox(H_est)) << "H: " << H << "\n H_est: " << H_est;

  Eigen::aligned_vector<Sophus::SE3d> motion_hypothesis =
      granite::decompose_homography(H);

  ASSERT_EQ(motion_hypothesis.size(), 8)
      << "decompose_homography should yield 8 motion hyptothesis, but "
      << motion_hypothesis.size() << " were returned";

  bool right_rot_found = false;
  Sophus::SE3d T_cama_camb_est;
  for (const auto& mot_hyp : motion_hypothesis) {
    double angular_dist = T_cama_camb.unit_quaternion().angularDistance(
        mot_hyp.unit_quaternion());

    if (angular_dist < 1e-5) {
      T_cama_camb_est = mot_hyp;
      right_rot_found = true;
    }
  }

  ASSERT_TRUE(right_rot_found)
      << "Right rotation was not amongst the motion hypothesis";

  // estimate scale of translation
  double scale = 1.0;
  const Eigen::Vector3d t_cama_camb = T_cama_camb.translation();
  Eigen::Vector3d t_cama_camb_est = T_cama_camb_est.translation();

  for (size_t idx = 0; idx < 3; idx++) {
    if (fabs(t_cama_camb(idx)) > 1e1 * std::numeric_limits<double>::epsilon() &&
        fabs(t_cama_camb_est(idx)) >
            1e1 * std::numeric_limits<double>::epsilon()) {
      scale = t_cama_camb(idx) / t_cama_camb_est(idx);
      break;
    }
  }
  t_cama_camb_est *= scale;

  ASSERT_TRUE(t_cama_camb.isApprox(t_cama_camb_est))
      << "The translations do not match!\n"
      << "t_cama_camb: " << t_cama_camb
      << "\n t_cama_camb_est: " << t_cama_camb_est;

  // check filtering
  Eigen::aligned_vector<Sophus::SE3d> filtered_motion_hypothesis =
      granite::filter_motion_hypothesis(motion_hypothesis, bearings_a,
                                       bearings_b);

  std::cout << "Nr hypothesis after filtering: "
            << filtered_motion_hypothesis.size() << std::endl;

  right_rot_found = false;
  for (const auto& mot_hyp : filtered_motion_hypothesis) {
    double angular_dist = T_cama_camb.unit_quaternion().angularDistance(
        mot_hyp.unit_quaternion());

    if (angular_dist < 1e-5) {
      T_cama_camb_est = mot_hyp;
      right_rot_found = true;
    }
  }

  ASSERT_TRUE(right_rot_found)
      << "Right rotation was not amongst the filtered motion hypothesis";

  // estimate scale of translation
  scale = 1.0;
  t_cama_camb_est = T_cama_camb_est.translation();

  for (size_t idx = 0; idx < 3; idx++) {
    if (fabs(t_cama_camb(idx)) > 1e1 * std::numeric_limits<double>::epsilon() &&
        fabs(t_cama_camb_est(idx)) >
            1e1 * std::numeric_limits<double>::epsilon()) {
      scale = t_cama_camb(idx) / t_cama_camb_est(idx);
      break;
    }
  }
  t_cama_camb_est *= scale;

  ASSERT_TRUE(t_cama_camb.isApprox(t_cama_camb_est))
      << "The translations do not match!\n"
      << "t_cama_camb: " << t_cama_camb
      << "\n t_cama_camb_est: " << t_cama_camb_est;
}
