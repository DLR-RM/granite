

#include <granite/optimization/accumulator.h>
#include <granite/spline/se3_spline.h>
#include <granite/vi_estimator/ba_base.h>
#include <granite/vi_estimator/keypoint_vio.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

static const double accel_std_dev = 0.23;
static const double gyro_std_dev = 0.0027;

// Smaller noise for testing
// static const double accel_std_dev = 0.00023;
// static const double gyro_std_dev = 0.0000027;

std::random_device rd{};
std::mt19937 gen{rd()};

std::normal_distribution<> gyro_noise_dist{0, gyro_std_dev};
std::normal_distribution<> accel_noise_dist{0, accel_std_dev};

TEST(VioTestSuite, ImuNullspace2Test) {
  int num_knots = 15;

  Eigen::Vector3d bg, ba;
  bg = Eigen::Vector3d::Random() / 100;
  ba = Eigen::Vector3d::Random() / 10;

  granite::IntegratedImuMeasurement imu_meas(0, bg, ba);

  granite::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  granite::PoseVelBiasState state0, state1, state1_gt;

  state0.t_ns = 0;
  state0.T_w_i = gt_spline.pose(int64_t(0));
  state0.vel_w_i = gt_spline.transVelWorld(int64_t(0));
  state0.bias_gyro = bg;
  state0.bias_accel = ba;

  Eigen::Vector3d accel_cov, gyro_cov;
  accel_cov.setConstant(accel_std_dev * accel_std_dev);
  gyro_cov.setConstant(gyro_std_dev * gyro_std_dev);

  int64_t dt_ns = 1e7;
  for (int64_t t_ns = dt_ns / 2;
       t_ns < int64_t(1e8);  //  gt_spline.maxTimeNs() - int64_t(1e9);
       t_ns += dt_ns) {
    Sophus::SE3d pose = gt_spline.pose(t_ns);
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - granite::constants::g);
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    granite::ImuData data;
    data.accel = accel_body + ba;
    data.gyro = rot_vel_body + bg;

    data.accel[0] += accel_noise_dist(gen);
    data.accel[1] += accel_noise_dist(gen);
    data.accel[2] += accel_noise_dist(gen);

    data.gyro[0] += gyro_noise_dist(gen);
    data.gyro[1] += gyro_noise_dist(gen);
    data.gyro[2] += gyro_noise_dist(gen);

    data.t_ns = t_ns + dt_ns / 2;  // measurement in the middle of the interval;

    imu_meas.integrate(data, accel_cov, gyro_cov);
  }

  state1.t_ns = imu_meas.get_dt_ns();
  state1.T_w_i = gt_spline.pose(imu_meas.get_dt_ns()) *
                 Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  state1.vel_w_i = gt_spline.transVelWorld(imu_meas.get_dt_ns()) +
                   Sophus::Vector3d::Random() / 10;
  state1.bias_gyro = bg;
  state1.bias_accel = ba;

  Eigen::Vector3d gyro_weight;
  gyro_weight.setConstant(1e6);

  Eigen::Vector3d accel_weight;
  accel_weight.setConstant(1e6);

  Eigen::aligned_map<int64_t, granite::IntegratedImuMeasurement> imu_meas_vec;
  Eigen::aligned_map<int64_t, granite::PoseVelBiasStateWithLin> frame_states;
  Eigen::aligned_map<int64_t, granite::PoseStateWithLin> frame_poses;

  imu_meas_vec[state0.t_ns] = imu_meas;
  frame_states[state0.t_ns] = state0;
  frame_states[state1.t_ns] = state1;

  int asize = 30;
  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  H.setZero(asize, asize);
  b.setZero(asize);

  granite::AbsOrderMap aom;
  aom.total_size = 30;
  aom.items = 2;
  aom.abs_order_map[state0.t_ns] = std::make_pair(0, 15);
  aom.abs_order_map[state1.t_ns] = std::make_pair(15, 15);

  double imu_error, bg_error, ba_error;
  granite::KeypointVioEstimator::linearizeAbsIMU(
      aom, H, b, imu_error, bg_error, ba_error, frame_states, imu_meas_vec,
      gyro_weight, accel_weight, granite::constants::g);

  // Check quadratic approximation
  for (int i = 0; i < 10; i++) {
    Eigen::VectorXd rand_inc;
    rand_inc.setRandom(asize);
    rand_inc.normalize();
    rand_inc /= 10000;

    auto frame_states_copy = frame_states;
    frame_states_copy[state0.t_ns].applyInc(rand_inc.segment<15>(0));
    frame_states_copy[state1.t_ns].applyInc(rand_inc.segment<15>(15));

    double imu_error_u, bg_error_u, ba_error_u;
    granite::KeypointVioEstimator::computeImuError(
        aom, imu_error_u, bg_error_u, ba_error_u, frame_states_copy,
        imu_meas_vec, gyro_weight, accel_weight, granite::constants::g);

    double e0 = imu_error + bg_error + ba_error;
    double e1 = imu_error_u + bg_error_u + ba_error_u - e0;

    double e2 = 0.5 * rand_inc.transpose() * H * rand_inc;
    e2 += rand_inc.transpose() * b;

    EXPECT_LE(std::abs(e1 - e2), 2e-2) << "e1 " << e1 << " e2 " << e2;
  }

  std::cout << "=========================================" << std::endl;
  Eigen::VectorXd null_res = granite::KeypointVioEstimator::checkNullspace(
      H, b, aom, frame_states, frame_poses);
  std::cout << "=========================================" << std::endl;

  EXPECT_LE(std::abs(null_res[0]), 1e-8);
  EXPECT_LE(std::abs(null_res[1]), 1e-8);
  EXPECT_LE(std::abs(null_res[2]), 1e-8);
  EXPECT_LE(std::abs(null_res[5]), 1e-6);
}

TEST(VioTestSuite, ImuNullspace3Test) {
  int num_knots = 15;

  Eigen::Vector3d bg, ba;
  bg = Eigen::Vector3d::Random() / 100;
  ba = Eigen::Vector3d::Random() / 10;

  granite::IntegratedImuMeasurement imu_meas1(0, bg, ba);

  granite::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  granite::PoseVelBiasState state0, state1, state2;

  state0.t_ns = 0;
  state0.T_w_i = gt_spline.pose(int64_t(0));
  state0.vel_w_i = gt_spline.transVelWorld(int64_t(0));
  state0.bias_gyro = bg;
  state0.bias_accel = ba;

  Eigen::Vector3d accel_cov, gyro_cov;
  accel_cov.setConstant(accel_std_dev * accel_std_dev);
  gyro_cov.setConstant(gyro_std_dev * gyro_std_dev);

  int64_t dt_ns = 1e7;
  for (int64_t t_ns = dt_ns / 2;
       t_ns < int64_t(1e9);  //  gt_spline.maxTimeNs() - int64_t(1e9);
       t_ns += dt_ns) {
    Sophus::SE3d pose = gt_spline.pose(t_ns);
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - granite::constants::g);
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    granite::ImuData data;
    data.accel = accel_body + ba;
    data.gyro = rot_vel_body + bg;

    data.accel[0] += accel_noise_dist(gen);
    data.accel[1] += accel_noise_dist(gen);
    data.accel[2] += accel_noise_dist(gen);

    data.gyro[0] += gyro_noise_dist(gen);
    data.gyro[1] += gyro_noise_dist(gen);
    data.gyro[2] += gyro_noise_dist(gen);

    data.t_ns = t_ns + dt_ns / 2;  // measurement in the middle of the interval;

    imu_meas1.integrate(data, accel_cov, gyro_cov);
  }

  granite::IntegratedImuMeasurement imu_meas2(imu_meas1.get_dt_ns(), bg, ba);
  for (int64_t t_ns = imu_meas1.get_dt_ns() + dt_ns / 2;
       t_ns < int64_t(2e9);  //  gt_spline.maxTimeNs() - int64_t(1e9);
       t_ns += dt_ns) {
    Sophus::SE3d pose = gt_spline.pose(t_ns);
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - granite::constants::g);
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    granite::ImuData data;
    data.accel = accel_body + ba;
    data.gyro = rot_vel_body + bg;

    data.accel[0] += accel_noise_dist(gen);
    data.accel[1] += accel_noise_dist(gen);
    data.accel[2] += accel_noise_dist(gen);

    data.gyro[0] += gyro_noise_dist(gen);
    data.gyro[1] += gyro_noise_dist(gen);
    data.gyro[2] += gyro_noise_dist(gen);

    data.t_ns = t_ns + dt_ns / 2;  // measurement in the middle of the interval;

    imu_meas2.integrate(data, accel_cov, gyro_cov);
  }

  state1.t_ns = imu_meas1.get_dt_ns();
  state1.T_w_i = gt_spline.pose(state1.t_ns) *
                 Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  state1.vel_w_i =
      gt_spline.transVelWorld(state1.t_ns) + Sophus::Vector3d::Random() / 10;
  state1.bias_gyro = bg;
  state1.bias_accel = ba;

  state2.t_ns = imu_meas1.get_dt_ns() + imu_meas2.get_dt_ns();
  state2.T_w_i = gt_spline.pose(state2.t_ns) *
                 Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  state2.vel_w_i =
      gt_spline.transVelWorld(state2.t_ns) + Sophus::Vector3d::Random() / 10;
  state2.bias_gyro = bg;
  state2.bias_accel = ba;

  Eigen::Vector3d gyro_weight;
  gyro_weight.setConstant(1e6);

  Eigen::Vector3d accel_weight;
  accel_weight.setConstant(1e6);

  Eigen::aligned_map<int64_t, granite::IntegratedImuMeasurement> imu_meas_vec;
  Eigen::aligned_map<int64_t, granite::PoseVelBiasStateWithLin> frame_states;
  Eigen::aligned_map<int64_t, granite::PoseStateWithLin> frame_poses;

  imu_meas_vec[imu_meas1.get_start_t_ns()] = imu_meas1;
  imu_meas_vec[imu_meas2.get_start_t_ns()] = imu_meas2;
  frame_states[state0.t_ns] = state0;
  frame_states[state1.t_ns] = state1;
  frame_states[state2.t_ns] = state2;

  int asize = 45;
  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  H.setZero(asize, asize);
  b.setZero(asize);

  granite::AbsOrderMap aom;
  aom.total_size = asize;
  aom.items = 2;
  aom.abs_order_map[state0.t_ns] = std::make_pair(0, 15);
  aom.abs_order_map[state1.t_ns] = std::make_pair(15, 15);
  aom.abs_order_map[state2.t_ns] = std::make_pair(30, 15);

  double imu_error, bg_error, ba_error;
  granite::KeypointVioEstimator::linearizeAbsIMU(
      aom, H, b, imu_error, bg_error, ba_error, frame_states, imu_meas_vec,
      gyro_weight, accel_weight, granite::constants::g);

  std::cout << "=========================================" << std::endl;
  Eigen::VectorXd null_res = granite::KeypointVioEstimator::checkNullspace(
      H, b, aom, frame_states, frame_poses);
  std::cout << "=========================================" << std::endl;

  EXPECT_LE(std::abs(null_res[0]), 1e-8);
  EXPECT_LE(std::abs(null_res[1]), 1e-8);
  EXPECT_LE(std::abs(null_res[2]), 1e-8);
  EXPECT_LE(std::abs(null_res[5]), 1e-6);
}

TEST(VioTestSuite, RelPoseTest) {
  Sophus::SE3d T_w_i_h = Sophus::se3_expd(Sophus::Vector6d::Random());
  Sophus::SE3d T_w_i_t = Sophus::se3_expd(Sophus::Vector6d::Random());

  Sophus::SE3d T_i_c_h = Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  Sophus::SE3d T_i_c_t = Sophus::se3_expd(Sophus::Vector6d::Random() / 10);

  Sophus::Matrix6d d_rel_d_h, d_rel_d_t;

  Sophus::SE3d T_t_h_sophus = granite::KeypointVioEstimator::computeRelPose(
      T_w_i_h, T_i_c_h, T_w_i_t, T_i_c_t, &d_rel_d_h, &d_rel_d_t);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_h", d_rel_d_h,
        [&](const Sophus::Vector6d& x) {
          Sophus::SE3d T_w_h_new = T_w_i_h;
          granite::PoseState::incPose(x, T_w_h_new);

          Sophus::SE3d T_t_h_sophus_new =
              granite::KeypointVioEstimator::computeRelPose(T_w_h_new, T_i_c_h,
                                                           T_w_i_t, T_i_c_t);

          return Sophus::se3_logd(T_t_h_sophus_new * T_t_h_sophus.inverse());
        },
        x0);
  }

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_t", d_rel_d_t,
        [&](const Sophus::Vector6d& x) {
          Sophus::SE3d T_w_t_new = T_w_i_t;
          granite::PoseState::incPose(x, T_w_t_new);

          Sophus::SE3d T_t_h_sophus_new =
              granite::KeypointVioEstimator::computeRelPose(T_w_i_h, T_i_c_h,
                                                           T_w_t_new, T_i_c_t);
          return Sophus::se3_logd(T_t_h_sophus_new * T_t_h_sophus.inverse());
        },
        x0);
  }
}

/*
TEST(VioTestSuite, RelPoseSim3Test) {
  const Sophus::Sim3d T_w_i_h = Sophus::sim3_expd(Sophus::Vector7d::Random());
  const Sophus::Sim3d T_w_i_t = Sophus::sim3_expd(Sophus::Vector7d::Random());

  const Sophus::SE3d T_i_c_h =
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  const Sophus::SE3d T_i_c_t =
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);

  Sophus::Matrix7d d_rel_d_h, d_rel_d_t;

  Sophus::Sim3d T_t_h_sophus = granite::KeypointVioEstimator::computeRelPoseSim3(
      T_w_i_h, T_i_c_h, T_w_i_t, T_i_c_t, &d_rel_d_h, &d_rel_d_t);

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_h", d_rel_d_h,
        [&](const Sophus::Vector7d& x) {
          Sophus::Sim3d T_w_h_new = T_w_i_h;
          granite::PoseState::incPose(x, T_w_h_new);

          Sophus::Sim3d T_t_h_sophus_new =
              granite::KeypointVioEstimator::computeRelPoseSim3(
                  T_w_h_new, T_i_c_h, T_w_i_t, T_i_c_t);

          return Sophus::sim3_logd(T_t_h_sophus_new * T_t_h_sophus.inverse());
        },
        x0);
  }

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_t", d_rel_d_t,
        [&](const Sophus::Vector7d& x) {
          Sophus::Sim3d T_w_t_new = T_w_i_t;
          granite::PoseState::incPose(x, T_w_t_new);

          Sophus::Sim3d T_t_h_sophus_new =
              granite::KeypointVioEstimator::concatRelPoseSim3(
                  T_w_i_h, T_i_c_h, T_w_t_new, T_i_c_t);
          return Sophus::sim3_logd(T_t_h_sophus_new * T_t_h_sophus.inverse());
        },
        x0);
  }
}
*/

TEST(VioTestSuite, ConcatRelPoseSE3Test) {
  const Sophus::SE3d T_ti_bi = Sophus::se3_expd(Sophus::Vector6d::Random());
  const Sophus::SE3d T_bi_hi = Sophus::se3_expd(Sophus::Vector6d::Random());

  const Sophus::SE3d T_i_c_h =
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  const Sophus::SE3d T_i_c_t =
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);

  const Eigen::aligned_vector<Sophus::SE3d> T_ti_his = {T_ti_bi, T_bi_hi};

  Eigen::aligned_vector<Sophus::Matrix6d> d_rel_d_xi(2);
  Eigen::aligned_vector<Sophus::Matrix6d> d_rel_d_xi_inv(2);

  Sophus::SE3d T_t_h = granite::KeypointVioEstimator::concatRelPoseSE3(
      T_ti_his, T_i_c_h, T_i_c_t, &d_rel_d_xi, &d_rel_d_xi_inv);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_xi 0", d_rel_d_xi.at(0),
        [&](const Sophus::Vector6d& x) {
          Sophus::SE3d T_ti_bi_new = T_ti_bi;
          granite::PoseState::incPose(x, T_ti_bi_new);

          Sophus::SE3d T_t_h_sophus_new =
              T_i_c_t.inverse() * T_ti_bi_new * T_bi_hi * T_i_c_h;

          return Sophus::se3_logd(T_t_h_sophus_new * T_t_h.inverse());
        },
        x0);
  }

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_xi 1", d_rel_d_xi.at(1),
        [&](const Sophus::Vector6d& x) {
          Sophus::SE3d T_bi_hi_new = T_bi_hi;
          granite::PoseState::incPose(x, T_bi_hi_new);

          Sophus::SE3d T_t_h_sophus_new =
              T_i_c_t.inverse() * T_ti_bi * T_bi_hi_new * T_i_c_h;

          return Sophus::se3_logd(T_t_h_sophus_new * T_t_h.inverse());
        },
        x0);
  }

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_xi_inv 0", d_rel_d_xi_inv.at(0),
        [&](const Sophus::Vector6d& x) {
          Sophus::SE3d T_ti_bi_new = T_ti_bi;
          granite::PoseState::incPose(x, T_ti_bi_new);

          Sophus::SE3d T_h_t_new = T_i_c_h.inverse() * T_bi_hi.inverse() *
                                   T_ti_bi_new.inverse() * T_i_c_t;

          return Sophus::se3_logd(T_h_t_new * T_t_h);
        },
        x0);
  }

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_xi_inv 1", d_rel_d_xi_inv.at(1),
        [&](const Sophus::Vector6d& x) {
          Sophus::SE3d T_bi_hi_new = T_bi_hi;
          granite::PoseState::incPose(x, T_bi_hi_new);

          Sophus::SE3d T_h_t_new = T_i_c_h.inverse() * T_bi_hi_new.inverse() *
                                   T_ti_bi.inverse() * T_i_c_t;

          return Sophus::se3_logd(T_h_t_new * T_t_h);
        },
        x0);
  }
}

TEST(VioTestSuite, ConcatRelPoseSim3Test) {
  const Sophus::Sim3d T_ti_bi = Sophus::sim3_expd(Sophus::Vector7d::Random());
  const Sophus::Sim3d T_bi_hi = Sophus::sim3_expd(Sophus::Vector7d::Random());

  const Sophus::SE3d T_i_c_h =
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  const Sophus::SE3d T_i_c_t =
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);

  const Eigen::aligned_vector<Sophus::Sim3d> T_ti_his = {T_ti_bi, T_bi_hi};

  Eigen::aligned_vector<Sophus::Matrix7d> d_rel_d_xi(2);
  Eigen::aligned_vector<Sophus::Matrix7d> d_rel_d_xi_inv(2);

  Sophus::Sim3d T_t_h = granite::KeypointVioEstimator::concatRelPoseSim3(
      T_ti_his, T_i_c_h, T_i_c_t, &d_rel_d_xi, &d_rel_d_xi_inv);

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_xi 0", d_rel_d_xi.at(0),
        [&](const Sophus::Vector7d& x) {
          Sophus::Sim3d T_ti_bi_new = T_ti_bi;
          granite::PoseState::incPose(x, T_ti_bi_new);

          Sophus::Sim3d T_t_h_sophus_new =
              Sophus::se3_2_sim3(T_i_c_t.inverse()) * T_ti_bi_new * T_bi_hi *
              Sophus::se3_2_sim3(T_i_c_h);

          return Sophus::sim3_logd(T_t_h_sophus_new * T_t_h.inverse());
        },
        x0);
  }

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_xi 1", d_rel_d_xi.at(1),
        [&](const Sophus::Vector7d& x) {
          Sophus::Sim3d T_bi_hi_new = T_bi_hi;
          granite::PoseState::incPose(x, T_bi_hi_new);

          Sophus::Sim3d T_t_h_sophus_new =
              Sophus::se3_2_sim3(T_i_c_t.inverse()) * T_ti_bi * T_bi_hi_new *
              Sophus::se3_2_sim3(T_i_c_h);

          return Sophus::sim3_logd(T_t_h_sophus_new * T_t_h.inverse());
        },
        x0);
  }

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_xi_inv 0", d_rel_d_xi_inv.at(0),
        [&](const Sophus::Vector7d& x) {
          Sophus::Sim3d T_ti_bi_new = T_ti_bi;
          granite::PoseState::incPose(x, T_ti_bi_new);

          Sophus::Sim3d T_h_t_new = Sophus::se3_2_sim3(T_i_c_h.inverse()) *
                                    T_bi_hi.inverse() * T_ti_bi_new.inverse() *
                                    Sophus::se3_2_sim3(T_i_c_t);

          return Sophus::sim3_logd(T_h_t_new * T_t_h);
        },
        x0);
  }

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_jacobian(
        "d_rel_d_xi_inv 1", d_rel_d_xi_inv.at(1),
        [&](const Sophus::Vector7d& x) {
          Sophus::Sim3d T_bi_hi_new = T_bi_hi;
          granite::PoseState::incPose(x, T_bi_hi_new);

          Sophus::Sim3d T_h_t_new = Sophus::se3_2_sim3(T_i_c_h.inverse()) *
                                    T_bi_hi_new.inverse() * T_ti_bi.inverse() *
                                    Sophus::se3_2_sim3(T_i_c_t);

          return Sophus::sim3_logd(T_h_t_new * T_t_h);
        },
        x0);
  }
}

TEST(VioTestSuite, LinearizePointsTest) {
  granite::ExtendedUnifiedCamera<double> cam =
      granite::ExtendedUnifiedCamera<double>::getTestProjections()[0];

  granite::KeypointPosition kpt_pos;

  Eigen::Vector4d point3d;
  cam.unproject(Eigen::Vector2d::Random() * 50, point3d);
  kpt_pos.dir = granite::StereographicParam<double>::project(point3d);
  kpt_pos.id = 0.1231231;

  Sophus::SE3d T_w_h = Sophus::se3_expd(Sophus::Vector6d::Random() / 100);
  Sophus::SE3d T_w_t = Sophus::se3_expd(Sophus::Vector6d::Random() / 100);
  T_w_t.translation()[0] += 0.1;

  Sophus::SE3d T_t_h_sophus = T_w_t.inverse() * T_w_h;
  Eigen::Matrix4d T_t_h = T_t_h_sophus.matrix();

  Eigen::Vector4d p_trans;
  p_trans = granite::StereographicParam<double>::unproject(kpt_pos.dir);
  p_trans(3) = kpt_pos.id;

  p_trans = T_t_h * p_trans;

  granite::KeypointObservation kpt_obs;
  cam.project(p_trans, kpt_obs.pos);

  Eigen::Matrix<double, 5, 1> res;
  Eigen::Matrix<double, 5, 6> d_res_d_xi;
  Eigen::Matrix<double, 5, 3> d_res_d_p;

  const double regularizer_weight = 1e-6;

  granite::KeypointVioEstimator::linearizePoint(kpt_obs, kpt_pos, T_t_h, cam,
                                               res, regularizer_weight,
                                               &d_res_d_xi, &d_res_d_p);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_xi", d_res_d_xi,
        [&](const Sophus::Vector6d& x) {
          Eigen::Matrix4d T_t_h_new =
              (Sophus::se3_expd(x) * T_t_h_sophus).matrix();

          Eigen::Matrix<double, 5, 1> res;
          granite::KeypointVioEstimator::linearizePoint(
              kpt_obs, kpt_pos, T_t_h_new, cam, res, regularizer_weight);

          return res;
        },
        x0);
  }

  {
    Eigen::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_p", d_res_d_p,
        [&](const Eigen::Vector3d& x) {
          granite::KeypointPosition kpt_pos_new = kpt_pos;

          kpt_pos_new.dir += x.head<2>();
          kpt_pos_new.id += x[2];

          Eigen::Matrix<double, 5, 1> res;
          granite::KeypointVioEstimator::linearizePoint(
              kpt_obs, kpt_pos_new, T_t_h, cam, res, regularizer_weight);

          return res;
        },
        x0);
  }
}

TEST(VioTestSuite, LinearizeHelperSim3) {
  const Sophus::SE3d T_ai_bi = Sophus::se3_expd(Sophus::Vector6d::Random());
  const Sophus::SE3d T_bi_ci = Sophus::se3_expd(Sophus::Vector6d::Random());

  const Sophus::SE3d T_ci_cc =
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);
  const Sophus::SE3d T_ai_ac =
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);

  const Eigen::aligned_vector<Sophus::Sim3d> T_ti_his = {
      Sophus::se3_2_sim3(T_ai_bi), Sophus::se3_2_sim3(T_bi_ci)};

  Eigen::aligned_vector<Sophus::Matrix7d> d_rel_d_xi(2);
  Eigen::aligned_vector<Sophus::Matrix7d> d_rel_d_xi_inv(2);

  Sophus::Sim3d T_ac_cc = granite::BundleAdjustmentBase::concatRelPoseSim3(
      T_ti_his, T_ci_cc, T_ai_ac, &d_rel_d_xi, &d_rel_d_xi_inv);

  granite::ExtendedUnifiedCamera<double> cam =
      granite::ExtendedUnifiedCamera<double>::getTestProjections()[0];

  granite::KeypointPosition kpt_pos;

  Eigen::Vector4d point3d_hc;
  cam.unproject(Eigen::Vector2d::Random() * 50, point3d_hc);
  kpt_pos.dir = granite::StereographicParam<double>::project(point3d_hc);
  kpt_pos.id = 0.1231231;

  // c is host, a is target
  Eigen::Vector4d point3d_tc;
  point3d_tc = granite::StereographicParam<double>::unproject(kpt_pos.dir);
  point3d_tc(3) = kpt_pos.id;
  point3d_tc =
      (T_ai_ac.inverse() * T_ai_bi * T_bi_ci * T_ci_cc).matrix() * point3d_tc;

  granite::KeypointObservation kpt_obs;
  cam.project(point3d_tc, kpt_obs.pos);

  Eigen::Matrix<double, 5, 1> res;
  Eigen::Matrix<double, 5, granite::sim3_SIZE> d_res_d_xi;
  Eigen::Matrix<double, 5, 3> d_res_d_p;

  granite::BundleAdjustmentBase::linearizePoint(
      kpt_obs, kpt_pos, T_ac_cc, cam, res, 1e-12, &d_res_d_xi, &d_res_d_p);

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_jacobian(
        "d_T_ti_bi_d_xi", d_res_d_xi * d_rel_d_xi.at(1),
        [&](const Sophus::Vector7d& x) {
          Sophus::Sim3d T_bi_ci_new = Sophus::se3_2_sim3(T_bi_ci);
          granite::PoseState::incPose(x, T_bi_ci_new);

          Sophus::Sim3d T_ac_cc_new = Sophus::se3_2_sim3(T_ai_ac.inverse()) *
                                      Sophus::se3_2_sim3(T_ai_bi) *
                                      T_bi_ci_new * Sophus::se3_2_sim3(T_ci_cc);

          granite::BundleAdjustmentBase::linearizePoint(
              kpt_obs, kpt_pos, T_ac_cc_new, cam, res, 1e-12);

          return res;
        },
        x0);
  }

  // a is host, c is target
  point3d_tc = granite::StereographicParam<double>::unproject(kpt_pos.dir);
  point3d_tc(3) = kpt_pos.id;
  point3d_tc =
      (T_ci_cc.inverse() * T_bi_ci.inverse() * T_ai_bi.inverse() * T_ai_ac)
          .matrix() *
      point3d_tc;

  cam.project(point3d_tc, kpt_obs.pos);

  granite::BundleAdjustmentBase::linearizePoint(kpt_obs, kpt_pos,
                                               T_ac_cc.inverse(), cam, res,
                                               1e-12, &d_res_d_xi, &d_res_d_p);

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_jacobian(
        "d_T_ti_bi_d_xi inv", d_res_d_xi * d_rel_d_xi_inv.at(1),
        [&](const Sophus::Vector7d& x) {
          Sophus::Sim3d T_bi_ci_new = Sophus::se3_2_sim3(T_bi_ci);
          granite::PoseState::incPose(x, T_bi_ci_new);

          Sophus::Sim3d T_cc_ac_new = Sophus::se3_2_sim3(T_ci_cc.inverse()) *
                                      T_bi_ci_new.inverse() *
                                      Sophus::se3_2_sim3(T_ai_bi.inverse()) *
                                      Sophus::se3_2_sim3(T_ai_ac);

          granite::BundleAdjustmentBase::linearizePoint(
              kpt_obs, kpt_pos, T_cc_ac_new, cam, res, 1e-12);

          return res;
        },
        x0);
  }
}

TEST(VioTestSuite, LinearizePointsInfinityTest) {
  granite::ExtendedUnifiedCamera<double> cam =
      granite::ExtendedUnifiedCamera<double>::getTestProjections()[0];

  granite::KeypointPosition kpt_pos;

  Eigen::Vector4d point3d;
  cam.unproject(Eigen::Vector2d::Random() * 50, point3d);
  kpt_pos.dir = granite::StereographicParam<double>::project(point3d);
  kpt_pos.id = 0.0;

  Sophus::SE3d T_w_h = Sophus::se3_expd(Sophus::Vector6d::Random() / 100);
  Sophus::SE3d T_w_t =
      Sophus::SE3d(Sophus::SO3d::exp(Eigen::Vector3d::Random() / 100),
                   T_w_h.translation() + Eigen::Vector3d::Random() / 1000);

  Sophus::SE3d T_t_h_sophus = T_w_t.inverse() * T_w_h;
  Eigen::Matrix4d T_t_h = T_t_h_sophus.matrix();

  Eigen::Vector4d p_trans;
  p_trans = granite::StereographicParam<double>::unproject(kpt_pos.dir);
  p_trans(3) = kpt_pos.id;

  p_trans = T_t_h * p_trans;

  granite::KeypointObservation kpt_obs;
  cam.project(p_trans, kpt_obs.pos);

  Eigen::Matrix<double, 5, 1> res;
  Eigen::Matrix<double, 5, 6> d_res_d_xi;
  Eigen::Matrix<double, 5, 3> d_res_d_p;

  const double regularizer_weight = 1e-6;

  granite::KeypointVioEstimator::linearizePoint(kpt_obs, kpt_pos, T_t_h, cam,
                                               res, regularizer_weight,
                                               &d_res_d_xi, &d_res_d_p);

  {
    Sophus::Vector6d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_xi", d_res_d_xi,
        [&](const Sophus::Vector6d& x) {
          Eigen::Matrix4d T_t_h_new =
              (Sophus::se3_expd(x) * T_t_h_sophus).matrix();

          Eigen::Matrix<double, 5, 1> res;
          granite::KeypointVioEstimator::linearizePoint(
              kpt_obs, kpt_pos, T_t_h_new, cam, res, regularizer_weight);

          return res;
        },
        x0);
  }

  {
    Eigen::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_p", d_res_d_p,
        [&](const Eigen::Vector3d& x) {
          granite::KeypointPosition kpt_pos_new = kpt_pos;

          kpt_pos_new.dir += x.head<2>();
          kpt_pos_new.id += x[2];

          Eigen::Matrix<double, 5, 1> res;
          granite::KeypointVioEstimator::linearizePoint(
              kpt_obs, kpt_pos_new, T_t_h, cam, res, regularizer_weight);

          return res;
        },
        x0);
  }
}

TEST(VioTestSuite, LinearizePointsSim3Test) {
  granite::ExtendedUnifiedCamera<double> cam =
      granite::ExtendedUnifiedCamera<double>::getTestProjections()[0];

  granite::KeypointPosition kpt_pos;

  Eigen::Vector4d point3d;
  cam.unproject(Eigen::Vector2d::Random() * 50, point3d);
  kpt_pos.dir = granite::StereographicParam<double>::project(point3d);
  kpt_pos.id = 0.1231231;

  Sophus::Sim3d T_w_h = Sophus::sim3_expd(Sophus::Vector7d::Random() / 100);
  Sophus::Sim3d T_w_t = Sophus::sim3_expd(Sophus::Vector7d::Random() / 100);
  T_w_t.translation()[0] += 0.1;

  Sophus::Sim3d T_t_h = T_w_t.inverse() * T_w_h;

  Eigen::Vector4d p_trans;
  p_trans = granite::StereographicParam<double>::unproject(kpt_pos.dir);
  p_trans(3) = kpt_pos.id;

  p_trans = T_t_h.matrix() * p_trans;

  granite::KeypointObservation kpt_obs;
  cam.project(p_trans, kpt_obs.pos);

  Eigen::Matrix<double, 5, 1> res;
  Eigen::Matrix<double, 5, 7> d_res_d_xi;
  Eigen::Matrix<double, 5, 3> d_res_d_p;

  const double regularizer_weight = 1e-6;

  granite::KeypointVioEstimator::linearizePoint(kpt_obs, kpt_pos, T_t_h, cam,
                                               res, regularizer_weight,
                                               &d_res_d_xi, &d_res_d_p);

  {
    Sophus::Vector7d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_xi", d_res_d_xi,
        [&](const Sophus::Vector7d& x) {
          Sophus::Sim3d T_t_h_new = Sophus::sim3_expd(x) * T_t_h;

          Eigen::Matrix<double, 5, 1> res;
          granite::KeypointVioEstimator::linearizePoint(
              kpt_obs, kpt_pos, T_t_h_new, cam, res, regularizer_weight);

          return res;
        },
        x0);
  }

  {
    Eigen::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_p", d_res_d_p,
        [&](const Eigen::Vector3d& x) {
          granite::KeypointPosition kpt_pos_new = kpt_pos;

          kpt_pos_new.dir += x.head<2>();
          kpt_pos_new.id += x[2];

          Eigen::Matrix<double, 5, 1> res;
          granite::KeypointVioEstimator::linearizePoint(
              kpt_obs, kpt_pos_new, T_t_h, cam, res, regularizer_weight);

          return res;
        },
        x0);
  }
}

TEST(VioTestSuite, MarginalizationTest) {
  granite::Calibration<double> calib;
  calib.T_i_c.emplace_back(Sophus::SE3d());
  auto cam = granite::GenericCamera<double>::fromString("eucm");
  cam.variant = granite::ExtendedUnifiedCamera<double>::getTestProjections()[0];
  calib.intrinsics.emplace_back(cam);
  granite::BundleAdjustmentBase ba(calib, granite::VioConfig());

  Eigen::aligned_map<int64_t, granite::PoseStateWithLin> T_w_is;
  for (size_t i = 0; i < 4; i++) {
    T_w_is[i * 1e8] = granite::PoseStateWithLin(
        i * 1e8, Sophus::se3_expd(Sophus::Vector6d::Random()));
  }

  constexpr int NUM_POINTS = 1000;
  Eigen::aligned_vector<Eigen::Vector4d> gt_points(NUM_POINTS);
  for (int i = 0; i < NUM_POINTS; i++) {
    Eigen::Vector4d point;

    point.head<3>() = Eigen::Vector3d::Random().normalized() * 5.0;
    point(3) = 1.0;

    gt_points.at(i) = point;
  }

  for (size_t p_idx = 0; p_idx < gt_points.size(); p_idx++) {
    int num_obs = 0;
    for (const auto& pose_kv : T_w_is) {
      granite::TimeCamId tcid(pose_kv.first, 0);

      Eigen::Vector4d point_c = calib.T_i_c.at(0).inverse().matrix() *
                                pose_kv.second.getPoseLin().inverse().matrix() *
                                gt_points.at(p_idx);
      Eigen::Vector2d proj;
      bool success = cam.project(point_c, proj);
      // proj += Eigen::Vector2d::Random();
      if (success) {
        if (!ba.lmdb.landmarkExists(p_idx)) {
          granite::KeypointPosition kpt_pos;
          kpt_pos.kf_id = tcid;
          Eigen::Vector4d point_bearing;
          cam.unproject(proj, point_bearing);
          kpt_pos.dir =
              granite::StereographicParam<double>::project(point_bearing);
          kpt_pos.id = 1.0 / point_c.head<3>().norm();

          ba.lmdb.addLandmark(p_idx, kpt_pos);
        }

        granite::KeypointObservation kpt_obs;
        kpt_obs.kpt_id = p_idx;
        kpt_obs.pos = proj;
        ba.lmdb.addObservation(tcid, kpt_obs);
        num_obs++;
      }
    }
    if (num_obs == 1) {
      ba.lmdb.removeLandmark(p_idx);
    }
  }

  auto solve_rel = [&]() {
    double rld_error;
    granite::BundleAdjustmentBase::LinDataRelScale<granite::sim3_SIZE> ld = {};
    ba.linearizeHelperRelSim3(ld, T_w_is, ba.lmdb.getObservations(), rld_error);

    // std::cout << "H:\n" << ld.Hpp << std::endl;

    // schur complement

    ld.invert_landmark_hessians();

    for (const auto& kv_l_Hpls : ld.Hpl) {
      const auto lm_id = kv_l_Hpls.first;

      const auto& Hll_inv = ld.Hll.at(lm_id);

      for (const auto& kv_i : kv_l_Hpls.second) {
        const auto rel_pose_i = kv_i.first;
        const size_t rel_pose_i_start = rel_pose_i * granite::sim3_SIZE;

        const Eigen::Matrix<double, granite::sim3_SIZE, 3> Hpl_Hll_inv =
            kv_i.second * Hll_inv;

        ld.bp.segment<granite::sim3_SIZE>(rel_pose_i_start) -=
            Hpl_Hll_inv * ld.bl.at(lm_id);

        for (const auto& kv_j : kv_l_Hpls.second) {
          const auto rel_pose_j = kv_j.first;
          const size_t rel_pose_j_start = rel_pose_j * granite::sim3_SIZE;

          ld.Hpp.block<granite::sim3_SIZE, granite::sim3_SIZE>(
              rel_pose_i_start, rel_pose_j_start) -=
              Hpl_Hll_inv * kv_j.second.transpose();
        }
      }
    }

    ld.Hpp(0, 0) += 1e12;
    ld.Hpp(1, 1) += 1e12;
    ld.Hpp(2, 2) += 1e12;

    for (size_t i = 0; i < T_w_is.size() - 1; i++) {
      ld.Hpp(granite::sim3_SIZE * i + 6, granite::sim3_SIZE * i + 6) += 1;
    }

    std::cout << "H:\n" << ld.Hpp << std::endl;

    std::cout << "det: " << ld.Hpp.determinant() << std::endl;

    Eigen::VectorXd delta_x = ld.Hpp.ldlt().solve(ld.bp);

    std::cout << "delta_x: " << delta_x.transpose() << std::endl;

    return ld;
  };

  solve_rel();

  // abs formulation

  granite::AbsOrderMap aom;
  for (const auto& kv : T_w_is) {
    aom.abs_order_map[kv.first] =
        std::make_pair(aom.total_size, granite::se3_SIZE);

    aom.total_size += granite::se3_SIZE;
    aom.items++;
  }

  ba.frame_poses = T_w_is;
  double rld_error_abs;
  Eigen::aligned_vector<
      granite::BundleAdjustmentBase::RelLinData<granite::se3_SIZE>>
      rld_vec;
  ba.linearizeHelper(rld_vec, ba.lmdb.getObservations(), rld_error_abs);

  std::cout << "rld_vec: " << rld_vec.size() << std::endl;

  granite::BundleAdjustmentBase::LinearizeAbsReduce<
      granite::DenseAccumulator<double>, granite::se3_SIZE>
      lopt(aom);
  tbb::blocked_range<Eigen::aligned_vector<
      granite::BundleAdjustmentBase::RelLinData<granite::se3_SIZE>>::iterator>
      range(rld_vec.begin(), rld_vec.end());
  tbb::parallel_reduce(range, lopt);

  lopt.accum.getH()(0, 0) += 1e12;
  lopt.accum.getH()(1, 1) += 1e12;
  lopt.accum.getH()(2, 2) += 1e12;
  lopt.accum.getH()(3, 3) += 1e12;
  lopt.accum.getH()(4, 4) += 1e12;
  lopt.accum.getH()(5, 5) += 1e12;

  std::cout << "H_abs:\n" << lopt.accum.getH() << std::endl;

  lopt.accum.setup_solver();
  const Eigen::VectorXd delta_x_abs = lopt.accum.solve(nullptr);
  std::cout << "delta_x_abs: " << delta_x_abs.transpose() << std::endl;

  std::set<int> idx_to_keep, idx_to_marg;
  for (size_t i = 0; i < T_w_is.size(); i++) {
    if (i == 0) {
      for (size_t j = 0; j < granite::se3_SIZE; j++)
        idx_to_marg.emplace(i * granite::se3_SIZE + j);
    } else {
      for (size_t j = 0; j < granite::se3_SIZE; j++)
        idx_to_keep.emplace(i * granite::se3_SIZE + j);
    }
  }

  Eigen::MatrixXd H_marg;
  Eigen::VectorXd b_marg;
  ba.marginalizeHelper(lopt.accum.getH(), lopt.accum.getB(), idx_to_keep,
                       idx_to_marg, H_marg, b_marg);

  T_w_is.erase(0);
  ba.lmdb.removeFrame(0);

  Eigen::aligned_vector<Eigen::Vector4d> second_points(NUM_POINTS);
  for (int i = 0; i < NUM_POINTS; i++) {
    Eigen::Vector4d point;

    point.head<3>() = Eigen::Vector3d::Random().normalized() * 5.0;
    point(3) = 1.0;

    second_points.at(i) = point;
  }

  for (size_t p_idx = 0; p_idx < second_points.size(); p_idx++) {
    int num_obs = 0;
    for (const auto& pose_kv : T_w_is) {
      granite::TimeCamId tcid(pose_kv.first, 0);

      Eigen::Vector4d point_c = calib.T_i_c.at(0).inverse().matrix() *
                                pose_kv.second.getPoseLin().inverse().matrix() *
                                second_points.at(p_idx);
      Eigen::Vector2d proj;
      bool success = cam.project(point_c, proj);
      // proj += Eigen::Vector2d::Random();
      if (success) {
        if (!ba.lmdb.landmarkExists(p_idx + NUM_POINTS)) {
          granite::KeypointPosition kpt_pos;
          kpt_pos.kf_id = tcid;
          Eigen::Vector4d point_bearing;
          cam.unproject(proj, point_bearing);
          kpt_pos.dir =
              granite::StereographicParam<double>::project(point_bearing);
          kpt_pos.id = 1.0 / point_c.head<3>().norm();

          ba.lmdb.addLandmark(p_idx + NUM_POINTS, kpt_pos);
        }

        granite::KeypointObservation kpt_obs;
        kpt_obs.kpt_id = p_idx + NUM_POINTS;
        kpt_obs.pos = proj;
        ba.lmdb.addObservation(tcid, kpt_obs);

        num_obs++;
      }
    }
    if (num_obs == 1) {
      ba.lmdb.removeLandmark(p_idx + NUM_POINTS);
    }
  }

  auto ld = solve_rel();

  Eigen::aligned_vector<Sophus::SE3d> T_prev_next;
  for (auto iter = T_w_is.cbegin(); std::next(iter) != T_w_is.cend(); iter++) {
    T_prev_next.emplace_back(iter->second.getPoseLin().inverse() *
                             std::next(iter)->second.getPoseLin());
  }

  Eigen::aligned_vector<Sophus::Sim3d> chain_i;
  for (size_t i = 0; i < T_w_is.size(); i++) {
    int64_t i_t_ns = (i + 1) * 1e8;
    chain_i.emplace_back(Sophus::se3_2_sim3(T_w_is.at(i_t_ns).getPoseLin()));

    Eigen::aligned_vector<Sophus::Matrix7d> d_i_d_xi(chain_i.size());
    ba.concatRelPoseSim3(chain_i, Sophus::SE3d(), Sophus::SE3d(), &d_i_d_xi);

    Eigen::aligned_vector<Sophus::Sim3d> chain_j;
    for (size_t j = 0; j < T_w_is.size(); j++) {
      int64_t j_t_ns = (j + 1) * 1e8;
      chain_j.emplace_back(Sophus::se3_2_sim3(T_w_is.at(j_t_ns).getPoseLin()));

      Eigen::aligned_vector<Sophus::Matrix7d> d_j_d_xi(chain_j.size());
      ba.concatRelPoseSim3(chain_j, Sophus::SE3d(), Sophus::SE3d(), &d_j_d_xi);

      Sophus::Matrix7d H_marg_block_sim3;
      H_marg_block_sim3.setZero();
      H_marg_block_sim3.topLeftCorner<granite::se3_SIZE, granite::se3_SIZE>() =
          H_marg.block<granite::se3_SIZE, granite::se3_SIZE>(
              i * granite::se3_SIZE, j * granite::se3_SIZE);

      Sophus::Vector7d b_marg_segment_sim3;
      b_marg_segment_sim3.setZero();
      b_marg_segment_sim3.head<granite::se3_SIZE>() =
          b_marg.segment<granite::se3_SIZE>(i * granite::se3_SIZE);

      for (size_t ii = 0; ii < i; ii++) {
        ld.bp.segment<granite::sim3_SIZE>(ii * granite::sim3_SIZE) +=
            d_i_d_xi.at(ii + 1).transpose() * b_marg_segment_sim3;

        for (size_t jj = 0; jj < j; jj++) {
          ld.Hpp.block<granite::sim3_SIZE, granite::sim3_SIZE>(
              ii * granite::sim3_SIZE, jj * granite::sim3_SIZE) +=
              d_i_d_xi.at(ii + 1).transpose() * H_marg_block_sim3 *
              d_j_d_xi.at(jj + 1);
        }
      }
    }
  }

  Eigen::VectorXd delta_x_with_marg = ld.Hpp.ldlt().solve(ld.bp);

  std::cout << "delta_x_with_marg: " << delta_x_with_marg.transpose()
            << std::endl;
}