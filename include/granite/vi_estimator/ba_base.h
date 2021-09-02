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

#include <granite/utils/rel_pose_constraint.h>
#include <granite/vi_estimator/landmark_database.h>

#include <tbb/blocked_range.h>

namespace granite {

class BundleAdjustmentBase {
 public:
  BundleAdjustmentBase(const granite::Calibration<double>& calib,
                       const VioConfig& config);

  template <size_t POSE_SIZE>
  struct RelLinDataBase {
    std::vector<std::pair<TimeCamId, TimeCamId>> order;

    Eigen::aligned_vector<Eigen::Matrix<double, POSE_SIZE, POSE_SIZE>>
        d_rel_d_h;
    Eigen::aligned_vector<Eigen::Matrix<double, POSE_SIZE, POSE_SIZE>>
        d_rel_d_t;
  };

  template <size_t POSE_SIZE>
  struct FrameRelLinData {
    Eigen::Matrix<double, POSE_SIZE, POSE_SIZE> Hpp;
    Eigen::Matrix<double, POSE_SIZE, 1> bp;

    std::vector<int> lm_id;
    Eigen::aligned_vector<Eigen::Matrix<double, POSE_SIZE, 3>> Hpl;

    FrameRelLinData() {
      Hpp.setZero();
      bp.setZero();
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  template <size_t POSE_SIZE>
  struct RelLinData : public RelLinDataBase<POSE_SIZE> {
    RelLinData(size_t num_keypoints, size_t num_rel_poses) {
      Hll.reserve(num_keypoints);
      bl.reserve(num_keypoints);
      lm_to_obs.reserve(num_keypoints);

      Hpppl.reserve(num_rel_poses);
      this->order.reserve(num_rel_poses);

      this->d_rel_d_h.reserve(num_rel_poses);
      this->d_rel_d_t.reserve(num_rel_poses);

      error = 0;
    }

    void invert_keypoint_hessians() {
      for (auto& kv : Hll) {
        Eigen::Matrix3d Hll_inv;
        Hll_inv.setIdentity();
        kv.second.ldlt().solveInPlace(Hll_inv);
        kv.second = Hll_inv;
      }
    }

    Eigen::aligned_unordered_map<int, Eigen::Matrix3d> Hll;
    Eigen::aligned_unordered_map<int, Eigen::Vector3d> bl;
    Eigen::aligned_unordered_map<int, std::vector<std::pair<size_t, size_t>>>
        lm_to_obs;

    Eigen::aligned_vector<FrameRelLinData<POSE_SIZE>> Hpppl;

    double error;
  };

  template <size_t POSE_SIZE>
  struct LinDataRelScale {
    Eigen::aligned_unordered_map<int, Eigen::Matrix3d> Hll;
    Eigen::aligned_unordered_map<int, Eigen::Vector3d> bl;

    Eigen::MatrixXd Hpp;
    Eigen::VectorXd bp;

    // from lm_id -> (rel pose id -> Hpl)
    Eigen::aligned_unordered_map<
        int,
        Eigen::aligned_unordered_map<int, Eigen::Matrix<double, POSE_SIZE, 3>>>
        Hpl;

    void invert_landmark_hessians() {
      for (auto& kv : Hll) {
        Eigen::Matrix3d Hll_inv;
        Hll_inv.setIdentity();
        kv.second.ldlt().solveInPlace(Hll_inv);
        kv.second = Hll_inv;
      }
    }
  };

  void computeError(double& error,
                    std::map<int, std::vector<std::pair<TimeCamId, double>>>*
                        outliers = nullptr,
                    double outlier_threshold = 0) const;

  void linearizeHelper(
      Eigen::aligned_vector<RelLinData<se3_SIZE>>& rld_vec,
      const Eigen::aligned_map<
          TimeCamId,
          Eigen::aligned_map<TimeCamId,
                             Eigen::aligned_vector<KeypointObservation>>>&
          obs_to_lin,
      double& error) const;

  void linearizeHelperRelSE3(
      LinDataRelScale<se3_SIZE>& ld,
      const Eigen::aligned_map<int64_t, PoseStateWithLin>& kf_poses,
      const Eigen::aligned_map<
          TimeCamId,
          Eigen::aligned_map<TimeCamId,
                             Eigen::aligned_vector<KeypointObservation>>>&
          obs_to_lin,
      double& error) const;

  void linearizeHelperRelSim3(
      LinDataRelScale<sim3_SIZE>& ld,
      const Eigen::aligned_map<int64_t, PoseStateWithLin>& kf_poses,
      const Eigen::aligned_map<
          TimeCamId,
          Eigen::aligned_map<TimeCamId,
                             Eigen::aligned_vector<KeypointObservation>>>&
          obs_to_lin,
      double& error) const;

  template <size_t POSE_SIZE>
  static void linearizeRel(const RelLinData<POSE_SIZE>& rld, Eigen::MatrixXd& H,
                           Eigen::VectorXd& b) {
    //  std::cout << "linearizeRel: KF " << frame_states.size() << " obs "
    //            << obs.size() << std::endl;

    // Do schur complement
    size_t msize = rld.order.size();
    H.setZero(POSE_SIZE * msize, POSE_SIZE * msize);
    b.setZero(POSE_SIZE * msize);

    for (size_t i = 0; i < rld.order.size(); i++) {
      const FrameRelLinData<POSE_SIZE>& frld = rld.Hpppl.at(i);

      H.block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i, POSE_SIZE * i) += frld.Hpp;
      b.segment<POSE_SIZE>(POSE_SIZE * i) += frld.bp;

      for (size_t j = 0; j < frld.lm_id.size(); j++) {
        Eigen::Matrix<double, POSE_SIZE, 3> H_pl_H_ll_inv;
        int lm_id = frld.lm_id[j];

        H_pl_H_ll_inv = frld.Hpl[j] * rld.Hll.at(lm_id);
        b.segment<POSE_SIZE>(POSE_SIZE * i) -= H_pl_H_ll_inv * rld.bl.at(lm_id);

        const auto& other_obs = rld.lm_to_obs.at(lm_id);
        for (size_t k = 0; k < other_obs.size(); k++) {
          const FrameRelLinData<POSE_SIZE>& frld_other =
              rld.Hpppl.at(other_obs[k].first);
          int other_i = other_obs[k].first;

          Eigen::Matrix<double, 3, POSE_SIZE> H_l_p_other =
              frld_other.Hpl[other_obs[k].second].transpose();

          H.block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i, POSE_SIZE * other_i) -=
              H_pl_H_ll_inv * H_l_p_other;
        }
      }
    }
  }

  void filterOutliers(double outlier_threshold, size_t min_num_obs,
                      std::map<FrameId, int>* num_points_connected = nullptr);

  template <class CamT>
  static bool linearizePoint(
      const KeypointObservation& kpt_obs, const KeypointPosition& kpt_pos,
      const Eigen::Matrix4d& T_t_h, const CamT& cam,
      Eigen::Matrix<double, 5, 1>& res, double no_motion_regularizer_weight,
      Eigen::Matrix<double, 5, se3_SIZE>* d_res_d_xi = nullptr,
      Eigen::Matrix<double, 5, 3>* d_res_d_p = nullptr,
      Eigen::Vector4d* proj = nullptr) {
    // Todo implement without jacobians
    Eigen::Matrix<double, 4, 2> Jup;
    Eigen::Vector4d p_h_3d;
    p_h_3d = StereographicParam<double>::unproject(kpt_pos.dir, &Jup);
    p_h_3d[3] = kpt_pos.id;

    Eigen::Vector4d p_t_3d = T_t_h * p_h_3d;

    Eigen::Matrix<double, 4, se3_SIZE> d_point_d_xi;
    d_point_d_xi.topLeftCorner<3, 3>() =
        Eigen::Matrix3d::Identity() * kpt_pos.id;
    d_point_d_xi.topRightCorner<3, 3>() = -Sophus::SO3d::hat(p_t_3d.head<3>());
    d_point_d_xi.row(3).setZero();

    Eigen::Matrix<double, 2, 4> Jp;
    Eigen::Vector2d cam_proj;
    bool valid = cam.project(p_t_3d, cam_proj, &Jp);
    valid &= cam_proj.array().isFinite().all();

    if (!valid) {
      //      std::cerr << " Invalid projection! kpt_pos.dir "
      //                << kpt_pos.dir.transpose() << " kpt_pos.id " <<
      //                kpt_pos.id
      //                << " idx " << kpt_obs.kpt_id << std::endl;

      //      std::cerr << "T_t_h\n" << T_t_h << std::endl;
      //      std::cerr << "p_h_3d\n" << p_h_3d.transpose() << std::endl;
      //      std::cerr << "p_t_3d\n" << p_t_3d.transpose() << std::endl;

      return false;
    }

    if (proj) {
      proj->head<2>() = cam_proj;
      (*proj)[2] = p_t_3d[3] / p_t_3d.head<3>().norm();
    }
    res.head<2>() = cam_proj - kpt_obs.pos;

    // point at infinity
    if (kpt_pos.id == 0) {
      res.tail<3>() =
          no_motion_regularizer_weight * T_t_h.topRightCorner<3, 1>();
    } else {
      res.tail<3>().setZero();
    }

    if (d_res_d_xi) {
      d_res_d_xi->topLeftCorner<2, se3_SIZE>() = Jp * d_point_d_xi;

      d_res_d_xi->bottomLeftCorner<3, se3_SIZE>().setZero();
      // point at infinity
      if (kpt_pos.id == 0) {
        d_res_d_xi->bottomLeftCorner<3, 3>() =
            no_motion_regularizer_weight * Eigen::Matrix3d::Identity();
      }
    }

    if (d_res_d_p) {
      Eigen::Matrix<double, 4, 3> Jpp;
      Jpp.setZero();
      Jpp.block<3, 2>(0, 0) = T_t_h.topLeftCorner<3, 4>() * Jup;
      Jpp.col(2) = T_t_h.col(3);

      d_res_d_p->topLeftCorner<2, 3>() = Jp * Jpp;

      d_res_d_p->bottomLeftCorner<3, 3>().setZero();
      // point at infinity
      if (kpt_pos.id == 0) {
        d_res_d_p->col(2).setZero();
      }
    }

    return true;
  }

  template <class CamT>
  static bool linearizePoint(
      const KeypointObservation& kpt_obs, const KeypointPosition& kpt_pos,
      const Sophus::Sim3d& T_t_h, const CamT& cam,
      Eigen::Matrix<double, 5, 1>& res, double no_motion_regularizer_weight,
      Eigen::Matrix<double, 5, sim3_SIZE>* d_res_d_xi = nullptr,
      Eigen::Matrix<double, 5, 3>* d_res_d_p = nullptr,
      Eigen::Vector4d* proj = nullptr) {
    // Todo implement without jacobians
    Eigen::Matrix<double, 4, 2> Jup;
    Eigen::Vector4d p_h_3d;
    p_h_3d = StereographicParam<double>::unproject(kpt_pos.dir, &Jup);
    p_h_3d[3] = kpt_pos.id;

    Eigen::Vector4d p_t_3d = T_t_h.matrix() * p_h_3d;

    Eigen::Matrix<double, 4, sim3_SIZE> d_point_d_xi;
    d_point_d_xi.topLeftCorner<3, 3>() =
        Eigen::Matrix3d::Identity() * kpt_pos.id;
    d_point_d_xi.block<3, 3>(0, 3) = -Sophus::SO3d::hat(p_t_3d.head<3>());
    d_point_d_xi.col(6) = p_t_3d;
    d_point_d_xi.row(3).setZero();

    Eigen::Matrix<double, 2, 4> Jp;
    Eigen::Vector2d cam_proj;
    bool valid = cam.project(p_t_3d, cam_proj, &Jp);
    valid &= cam_proj.array().isFinite().all();

    if (!valid) {
      //      std::cerr << " Invalid projection! kpt_pos.dir "
      //                << kpt_pos.dir.transpose() << " kpt_pos.id " <<
      //                kpt_pos.id
      //                << " idx " << kpt_obs.kpt_id << std::endl;

      //      std::cerr << "T_t_h\n" << T_t_h << std::endl;
      //      std::cerr << "p_h_3d\n" << p_h_3d.transpose() << std::endl;
      //      std::cerr << "p_t_3d\n" << p_t_3d.transpose() << std::endl;

      return false;
    }

    if (proj) {
      proj->head<2>() = cam_proj;
      (*proj)[2] = p_t_3d[3] / p_t_3d.head<3>().norm();
    }
    res.head<2>() = cam_proj - kpt_obs.pos;

    // point at infinity
    if (kpt_pos.id == 0) {
      res.tail<3>() = no_motion_regularizer_weight * T_t_h.translation();
    } else {
      res.tail<3>().setZero();
    }

    if (d_res_d_xi) {
      d_res_d_xi->topLeftCorner<2, sim3_SIZE>() = Jp * d_point_d_xi;

      d_res_d_xi->bottomLeftCorner<3, sim3_SIZE>().setZero();
      // point at infinity
      if (kpt_pos.id == 0) {
        d_res_d_xi->bottomLeftCorner<3, 3>() =
            no_motion_regularizer_weight * Eigen::Matrix3d::Identity();
      }
    }

    if (d_res_d_p) {
      Eigen::Matrix<double, 4, 3> Jpp;
      Jpp.setZero();
      Jpp.block<3, 2>(0, 0) = T_t_h.matrix3x4() * Jup;
      Jpp.col(2).head<3>() = T_t_h.translation();
      Jpp.col(2)(3) = 1.0;

      d_res_d_p->topLeftCorner<2, 3>() = Jp * Jpp;

      d_res_d_p->bottomLeftCorner<3, 3>().setZero();
      // point at infinity
      if (kpt_pos.id == 0) {
        d_res_d_p->col(2).setZero();
      }
    }

    return true;
  }

  template <class CamT>
  inline static bool linearizePoint(
      const KeypointObservation& kpt_obs, const KeypointPosition& kpt_pos,
      const CamT& cam, Eigen::Vector2d& res,
      Eigen::Matrix<double, 2, 3>* d_res_d_p = nullptr,
      Eigen::Vector4d* proj = nullptr) {
    // Todo implement without jacobians
    Eigen::Matrix<double, 4, 2> Jup;
    Eigen::Vector4d p_h_3d;
    p_h_3d = StereographicParam<double>::unproject(kpt_pos.dir, &Jup);

    Eigen::Matrix<double, 2, 4> Jp;
    bool valid = cam.project(p_h_3d, res, &Jp);
    valid &= res.array().isFinite().all();

    if (!valid) {
      //      std::cerr << " Invalid projection! kpt_pos.dir "
      //                << kpt_pos.dir.transpose() << " kpt_pos.id " <<
      //                kpt_pos.id
      //                << " idx " << kpt_obs.kpt_id << std::endl;
      //      std::cerr << "p_h_3d\n" << p_h_3d.transpose() << std::endl;

      return false;
    }

    if (proj) {
      proj->head<2>() = res;
      (*proj)[2] = kpt_pos.id;
    }
    res -= kpt_obs.pos;

    if (d_res_d_p) {
      Eigen::Matrix<double, 4, 3> Jpp;
      Jpp.setZero();
      Jpp.block<4, 2>(0, 0) = Jup;
      Jpp.col(2).setZero();

      *d_res_d_p = Jp * Jpp;

      // point at infinity
      // if (kpt_pos.id == 0) {
      //   d_res_d_p->col(2).setZero();
      // }
    }

    return true;
  }

  void updatePoints(const AbsOrderMap& aom, const RelLinData<se3_SIZE>& rld,
                    const Eigen::VectorXd& inc);

  static Sophus::SE3d computeRelPose(const Sophus::SE3d& T_w_i_h,
                                     const Sophus::SE3d& T_w_i_t,
                                     const Sophus::SE3d& T_i_c_h,
                                     const Sophus::SE3d& T_i_c_t,
                                     Sophus::Matrix6d* d_rel_d_h = nullptr,
                                     Sophus::Matrix6d* d_rel_d_t = nullptr);

  // static Sophus::Sim3d computeRelPoseSim3(
  //     const Sophus::Sim3d& T_w_i_h, const Sophus::SE3d& T_i_c_h,
  //     const Sophus::Sim3d& T_w_i_t, const Sophus::SE3d& T_i_c_t,
  //     Sophus::Matrix7d* d_rel_d_h = nullptr,
  //     Sophus::Matrix7d* d_rel_d_t = nullptr);

  static Sophus::SE3d concatRelPoseSE3(
      const Eigen::aligned_vector<Sophus::SE3d>& T_ai_bis,
      const Sophus::SE3d& T_bi_bc, const Sophus::SE3d& T_ai_ac,
      Eigen::aligned_vector<Sophus::Matrix6d>* d_rel_d_xi = nullptr,
      Eigen::aligned_vector<Sophus::Matrix6d>* d_rel_d_xi_inv = nullptr);

  static Sophus::Sim3d concatRelPoseSim3(
      const Eigen::aligned_vector<Sophus::Sim3d>& T_ai_bis,
      const Sophus::SE3d& T_bi_bc, const Sophus::SE3d& T_ai_ac,
      Eigen::aligned_vector<Sophus::Matrix7d>* d_rel_d_xi = nullptr,
      Eigen::aligned_vector<Sophus::Matrix7d>* d_rel_d_xi_inv = nullptr);

  void get_current_points(Eigen::aligned_vector<Eigen::Vector3d>& points,
                          std::vector<int>& ids) const;

  // Modifies abs_H and abs_b as a side effect.
  static void marginalizeHelper(Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
                                const std::set<int>& idx_to_keep,
                                const std::set<int>& idx_to_marg,
                                Eigen::MatrixXd& marg_H,
                                Eigen::VectorXd& marg_b);

  void computeDelta(const AbsOrderMap& marg_order,
                    Eigen::VectorXd& delta) const;

  void linearizeMargPrior(const AbsOrderMap& marg_order,
                          const Eigen::MatrixXd& marg_H,
                          const Eigen::VectorXd& marg_b, const AbsOrderMap& aom,
                          Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
                          double& marg_prior_error) const;

  void computeMargPriorError(const AbsOrderMap& marg_order,
                             const Eigen::MatrixXd& marg_H,
                             const Eigen::VectorXd& marg_b,
                             double& marg_prior_error) const;

  static Eigen::VectorXd checkNullspace(
      const Eigen::MatrixXd& marg_H, const Eigen::VectorXd& marg_b,
      const AbsOrderMap& marg_order,
      const Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin>& frame_states,
      const Eigen::aligned_map<int64_t, PoseStateWithLin>& frame_poses);

  /// Triangulates the point and returns homogenous representation. First 3
  /// components - unit-length direction vector. Last component inverse
  /// distance.
  template <class Derived>
  static Eigen::Matrix<typename Derived::Scalar, 4, 1> triangulate(
      const Eigen::MatrixBase<Derived>& f0,
      const Eigen::MatrixBase<Derived>& f1,
      const Sophus::SE3<typename Derived::Scalar>& T_0_1) {
    EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);

    using Scalar = typename Derived::Scalar;
    using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

    Eigen::Matrix<Scalar, 3, 4> P1, P2;
    P1.setIdentity();
    P2 = T_0_1.inverse().matrix3x4();

    Eigen::Matrix<Scalar, 4, 4> A(4, 4);
    A.row(0) = f0[0] * P1.row(2) - f0[2] * P1.row(0);
    A.row(1) = f0[1] * P1.row(2) - f0[2] * P1.row(1);
    A.row(2) = f1[0] * P2.row(2) - f1[2] * P2.row(0);
    A.row(3) = f1[1] * P2.row(2) - f1[2] * P2.row(1);

    Eigen::JacobiSVD<Eigen::Matrix<Scalar, 4, 4>> mySVD(A, Eigen::ComputeFullV);
    Vec4 worldPoint = mySVD.matrixV().col(3);
    worldPoint /= worldPoint.template head<3>().norm();

    // Enforce same direction of bearing vector and initial point
    if (f0.dot(worldPoint.template head<3>()) < 0) worldPoint *= -1;

    return worldPoint;
  }

  template <class AccumT, size_t POSE_SIZE>
  static void linearizeAbs(const Eigen::MatrixXd& rel_H,
                           const Eigen::VectorXd& rel_b,
                           const RelLinDataBase<POSE_SIZE>& rld,
                           const AbsOrderMap& aom, AccumT& accum) {
    // int asize = aom.total_size;

    //  GRANITE_ASSERT(abs_H.cols() == asize);
    //  GRANITE_ASSERT(abs_H.rows() == asize);
    //  GRANITE_ASSERT(abs_b.rows() == asize);

    for (size_t i = 0; i < rld.order.size(); i++) {
      const TimeCamId& tcid_h = rld.order[i].first;
      const TimeCamId& tcid_ti = rld.order[i].second;

      int abs_h_idx = aom.abs_order_map.at(tcid_h.frame_id).first;
      if (aom.abs_order_map.count(tcid_ti.frame_id) == 0) continue;
      int abs_ti_idx = aom.abs_order_map.at(tcid_ti.frame_id).first;

      accum.template addB<POSE_SIZE>(
          abs_h_idx, rld.d_rel_d_h[i].transpose() *
                         rel_b.segment<POSE_SIZE>(i * POSE_SIZE));
      accum.template addB<POSE_SIZE>(
          abs_ti_idx, rld.d_rel_d_t[i].transpose() *
                          rel_b.segment<POSE_SIZE>(i * POSE_SIZE));

      for (size_t j = 0; j < rld.order.size(); j++) {
        GRANITE_ASSERT(rld.order[i].first == rld.order[j].first);

        const TimeCamId& tcid_tj = rld.order[j].second;

        if (aom.abs_order_map.count(tcid_tj.frame_id) == 0) continue;
        int abs_tj_idx = aom.abs_order_map.at(tcid_tj.frame_id).first;

        if (tcid_h.frame_id == tcid_ti.frame_id ||
            tcid_h.frame_id == tcid_tj.frame_id)
          continue;

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_h_idx, abs_h_idx,
            rld.d_rel_d_h[i].transpose() *
                rel_H.block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i,
                                                  POSE_SIZE * j) *
                rld.d_rel_d_h[j]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_ti_idx, abs_h_idx,
            rld.d_rel_d_t[i].transpose() *
                rel_H.block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i,
                                                  POSE_SIZE * j) *
                rld.d_rel_d_h[j]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_h_idx, abs_tj_idx,
            rld.d_rel_d_h[i].transpose() *
                rel_H.block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i,
                                                  POSE_SIZE * j) *
                rld.d_rel_d_t[j]);

        accum.template addH<POSE_SIZE, POSE_SIZE>(
            abs_ti_idx, abs_tj_idx,
            rld.d_rel_d_t[i].transpose() *
                rel_H.block<POSE_SIZE, POSE_SIZE>(POSE_SIZE * i,
                                                  POSE_SIZE * j) *
                rld.d_rel_d_t[j]);
      }
    }
  }

  static void linearizeRelTranslationConstraints(
      const AbsOrderMap& aom, Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b,
      double& error, const double weight,
      const std::unordered_map<FramePair, double>& rel_translation_constraints,
      const Eigen::aligned_map<int64_t, PoseStateWithLin>& poses) {
    error = 0;
    for (const auto& translation_constraint : rel_translation_constraints) {
      if (aom.abs_order_map.count(translation_constraint.first.frame_first) ==
              0 ||
          aom.abs_order_map.count(translation_constraint.first.frame_second) ==
              0)
        continue;

      const PoseStateWithLin& T_W_first =
          poses.at(translation_constraint.first.frame_first);
      const PoseStateWithLin& T_W_second =
          poses.at(translation_constraint.first.frame_second);

      Eigen::Matrix<double, 1, 3> d_res_d_T_w_first, d_res_d_T_w_second;
      double res = relTranslationError(
          translation_constraint.second, T_W_first.getPoseLin(),
          T_W_second.getPoseLin(), &d_res_d_T_w_first, &d_res_d_T_w_second);

      if (T_W_first.isLinearized() || T_W_second.isLinearized()) {
        res = relTranslationError(translation_constraint.second,
                                  T_W_first.getPose(), T_W_second.getPose());
      }

      error += 0.5 * weight * res * res;

      const size_t idx_first =
          aom.abs_order_map.at(translation_constraint.first.frame_first).first;
      const size_t idx_second =
          aom.abs_order_map.at(translation_constraint.first.frame_second).first;

      abs_H.block<3, 3>(idx_first, idx_first) +=
          weight * d_res_d_T_w_first.transpose() * d_res_d_T_w_first;
      abs_H.block<3, 3>(idx_first, idx_second) +=
          weight * d_res_d_T_w_first.transpose() * d_res_d_T_w_second;
      abs_H.block<3, 3>(idx_second, idx_first) +=
          weight * d_res_d_T_w_second.transpose() * d_res_d_T_w_first;
      abs_H.block<3, 3>(idx_second, idx_second) +=
          weight * d_res_d_T_w_second.transpose() * d_res_d_T_w_second;

      abs_b.segment<3>(idx_first) +=
          weight * d_res_d_T_w_first.transpose() * res;
      abs_b.segment<3>(idx_second) +=
          weight * d_res_d_T_w_second.transpose() * res;
    }
  }

  static void linearizeRelTranslationConstraintsRelSE3(
      Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b, double& error,
      const double weight,
      const std::unordered_map<FramePair, double>& rel_translation_constraints,
      const Eigen::aligned_map<int64_t, PoseStateWithLin>& poses) {
    error = 0;

    for (const auto& translation_constraint : rel_translation_constraints) {
      Eigen::aligned_vector<Sophus::SE3d> T_a_bs;
      std::vector<size_t> idxs;
      size_t idx = 0;
      for (auto iter = poses.cbegin(); std::next(iter) != poses.cend();
           iter++) {
        if (translation_constraint.first.frame_first >= iter->first &&
            translation_constraint.first.frame_second <= iter->first) {
          T_a_bs.emplace_back(std::next(iter)->second.getPoseLin().inverse() *
                              iter->second.getPoseLin());
          idxs.emplace_back(idx);
        }

        idx++;
      }
      Eigen::aligned_vector<Eigen::Matrix<double, 1, se3_SIZE>> d_res_d_xis(
          T_a_bs.size());

      const double res = relTranslationErrorSE3(translation_constraint.second,
                                                T_a_bs, &d_res_d_xis);

      error += 0.5 * weight * res * res;

      for (size_t i = 0; i < idxs.size(); i++) {
        const size_t i_start = idxs.at(i) * se3_SIZE;
        abs_b.segment<se3_SIZE>(i_start) +=
            weight * d_res_d_xis.at(i).transpose() * res;

        for (size_t j = 0; j < idxs.size(); j++) {
          const size_t j_start = idxs.at(j) * se3_SIZE;

          abs_H.block<se3_SIZE, se3_SIZE>(i_start, j_start) +=
              weight * d_res_d_xis.at(i).transpose() * d_res_d_xis.at(j);
        }
      }
    }
  }

  static void linearizeRelTranslationConstraintsRelSim3(
      Eigen::MatrixXd& abs_H, Eigen::VectorXd& abs_b, double& error,
      const double weight,
      const std::unordered_map<FramePair, double>& rel_translation_constraints,
      const Eigen::aligned_map<int64_t, PoseStateWithLin>& poses) {
    error = 0;

    for (const auto& translation_constraint : rel_translation_constraints) {
      Eigen::aligned_vector<Sophus::Sim3d> T_a_bs;
      std::vector<size_t> idxs;
      size_t idx = 0;
      for (auto iter = poses.cbegin(); std::next(iter) != poses.cend();
           iter++) {
        if (translation_constraint.first.frame_first >= iter->first &&
            translation_constraint.first.frame_second <= iter->first) {
          T_a_bs.emplace_back(Sophus::se3_2_sim3(
              std::next(iter)->second.getPoseLin().inverse() *
              iter->second.getPoseLin()));
          idxs.emplace_back(idx);
        }

        idx++;
      }
      Eigen::aligned_vector<Eigen::Matrix<double, 1, sim3_SIZE>> d_res_d_xis(
          T_a_bs.size());

      const double res = relTranslationErrorSim3(translation_constraint.second,
                                                 T_a_bs, &d_res_d_xis);

      error += 0.5 * weight * res * res;

      for (size_t i = 0; i < idxs.size(); i++) {
        const size_t i_start = idxs.at(i) * sim3_SIZE;
        abs_b.segment<sim3_SIZE>(i_start) +=
            weight * d_res_d_xis.at(i).transpose() * res;

        for (size_t j = 0; j < idxs.size(); j++) {
          const size_t j_start = idxs.at(j) * sim3_SIZE;

          abs_H.block<sim3_SIZE, sim3_SIZE>(i_start, j_start) +=
              weight * d_res_d_xis.at(i).transpose() * d_res_d_xis.at(j);
        }
      }
    }
  }

  static void computeRelTranslationConstraintsError(
      const AbsOrderMap& aom, double& error, const double weight,
      const std::unordered_map<FramePair, double>& rel_translation_constraints,
      const Eigen::aligned_map<int64_t, PoseStateWithLin>& poses) {
    error = 0;
    for (const auto& translation_constraint : rel_translation_constraints) {
      if (aom.abs_order_map.count(translation_constraint.first.frame_first) ==
              0 ||
          aom.abs_order_map.count(translation_constraint.first.frame_second) ==
              0)
        continue;

      const PoseStateWithLin T_W_first =
          poses.at(translation_constraint.first.frame_first);
      const PoseStateWithLin T_W_second =
          poses.at(translation_constraint.first.frame_second);

      const double res =
          relTranslationError(translation_constraint.second,
                              T_W_first.getPose(), T_W_second.getPose());
      error += 0.5 * weight * res * res;
    }
  }

  inline void eraseRelTranslationConstraints(FrameId id) {
    bool erased_one;
    do {
      auto rpc_it =
          std::find_if(rel_translation_constraints.begin(),
                       rel_translation_constraints.end(),
                       [id](const std::pair<FramePair, double>& t) -> bool {
                         return t.first.contains(id);
                       });
      erased_one = false;
      if (rpc_it != rel_translation_constraints.end()) {
        rel_translation_constraints.erase(rpc_it);
        erased_one = true;
      }
    } while (erased_one);
  }

  template <class AccumT, size_t POSE_SIZE>
  struct LinearizeAbsReduce {
    using RelLinDataIter =
        typename Eigen::aligned_vector<RelLinData<POSE_SIZE>>::iterator;

    LinearizeAbsReduce(AbsOrderMap& aom) : aom(aom) {
      accum.reset(aom.total_size);
    }

    LinearizeAbsReduce(const LinearizeAbsReduce& other, tbb::split)
        : aom(other.aom) {
      accum.reset(aom.total_size);
    }

    void operator()(const tbb::blocked_range<RelLinDataIter>& range) {
      for (RelLinData<POSE_SIZE>& rld : range) {
        rld.invert_keypoint_hessians();

        Eigen::MatrixXd rel_H;
        Eigen::VectorXd rel_b;
        linearizeRel<POSE_SIZE>(rld, rel_H, rel_b);

        linearizeAbs<AccumT, POSE_SIZE>(rel_H, rel_b, rld, aom, accum);
      }
    }

    void join(LinearizeAbsReduce& rhs) { accum.join(rhs.accum); }

    AbsOrderMap& aom;
    AccumT accum;
  };

  inline void backup() {
    for (auto& kv : frame_states) kv.second.backup();
    for (auto& kv : frame_poses) kv.second.backup();
    lmdb.backup();
  }

  inline void restore() {
    for (auto& kv : frame_states) kv.second.restore();
    for (auto& kv : frame_poses) kv.second.restore();
    lmdb.restore();
  }

  // protected:
  PoseStateWithLin getPoseStateWithLin(int64_t t_ns) const {
    auto it = frame_poses.find(t_ns);
    if (it != frame_poses.end()) return it->second;

    auto it2 = frame_states.find(t_ns);
    if (it2 == frame_states.end()) {
      std::cerr << "Could not find pose " << t_ns << std::endl;
      std::abort();
    }

    return PoseStateWithLin(it2->second);
  }

  Eigen::aligned_map<int64_t, PoseVelBiasStateWithLin> frame_states;
  Eigen::aligned_map<int64_t, PoseStateWithLin> frame_poses;

  // Point management
  LandmarkDatabase lmdb;

  granite::Calibration<double> calib;
  std::unordered_map<FramePair, double> rel_translation_constraints;

 protected:
  virtual void reset() {
    kf_ids.clear();
    num_points_kf.clear();
    lmdb.clear();
    rel_translation_constraints.clear();
    frame_states.clear();
    frame_poses.clear();
  }

  VioConfig config;
  std::set<int64_t> kf_ids;
  std::map<int64_t, int> num_points_kf;
};
}  // namespace granite
