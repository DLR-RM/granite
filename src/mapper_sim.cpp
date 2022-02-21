/**
MIT License

This file is part of the Granite project which is based on Basalt.
https://github.com/DLR-RM/granite

Copyright (c) Martin Wudenka, Deutsches Zentrum für Luft- und Raumfahrt

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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

#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <granite/io/dataset_io.h>
#include <granite/io/marg_data_io.h>
#include <granite/optimization/accumulator.h>
#include <granite/spline/se3_spline.h>
#include <granite/utils/filesystem.h>
#include <granite/utils/imu_types.h>
#include <granite/utils/nfr.h>
#include <granite/utils/sim_utils.h>
#include <granite/utils/vis_utils.h>
#include <granite/vi_estimator/keypoint_vio.h>
#include <granite/vi_estimator/nfr_mapper.h>
#include <granite/calibration/calibration.hpp>

#include <granite/serialization/headers_serialization.h>

using granite::se3_SIZE;
using granite::se3_VEL_BIAS_SIZE;

Eigen::Vector3d g(0, 0, -9.81);

std::shared_ptr<granite::Se3Spline<5>> gt_spline;

Eigen::aligned_vector<Sophus::SE3d> gt_frame_T_w_i;
Eigen::aligned_vector<Eigen::Vector3d> gt_frame_t_w_i, vio_t_w_i;
std::vector<int64_t> gt_frame_t_ns;

Eigen::aligned_vector<Eigen::Vector3d> gt_accel, gt_gyro, gt_accel_bias,
    gt_gyro_bias, noisy_accel, noisy_gyro, gt_vel;

std::vector<int64_t> gt_imu_t_ns;

Eigen::aligned_vector<Eigen::Vector3d> filter_points;
std::vector<int> filter_point_ids;

std::map<int64_t, granite::MargData::Ptr> marg_data;

Eigen::aligned_vector<granite::RollPitchFactor> roll_pitch_factors;
Eigen::aligned_vector<granite::RelPoseFactor> rel_pose_factors;

Eigen::aligned_vector<Eigen::Vector3d> edges_vis;
Eigen::aligned_vector<Eigen::Vector3d> roll_pitch_vis;
Eigen::aligned_vector<Eigen::Vector3d> rel_edges_vis;

Eigen::aligned_vector<Eigen::Vector3d> mapper_points;
std::vector<int> mapper_point_ids;

granite::NfrMapper::Ptr nrf_mapper;

std::map<granite::TimeCamId, granite::SimObservations> gt_observations;
std::map<granite::TimeCamId, granite::SimObservations> noisy_observations;

void draw_scene();
void load_data(const std::string& calib_path,
               const std::string& marg_data_path);
void processMargData(granite::MargData& m);
void extractNonlinearFactors(granite::MargData& m);
void computeEdgeVis();
void optimize();
void randomInc();
void randomYawInc();
double alignButton();
void setup_points();

constexpr int UI_WIDTH = 200;
// constexpr int NUM_FRAMES = 500;

granite::Calibration<double> calib;

pangolin::Var<bool> show_edges("ui.show_edges", true, false, true);
pangolin::Var<bool> show_points("ui.show_points", true, false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button optimize_btn("ui.optimize", &optimize);
Button rand_inc_btn("ui.rand_inc", &randomInc);
Button rand_yaw_inc_btn("ui.rand_yaw", &randomYawInc);
Button setup_points_btn("ui.setup_points", &setup_points);
Button align_se3_btn("ui.align_se3", &alignButton);

std::string marg_data_path;

int main(int argc, char** argv) {
  bool show_gui = true;
  std::string cam_calib_path;
  std::string result_path;

  CLI::App app{"App description"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--cam-calib", cam_calib_path,
                 "Ground-truth camera calibration used for simulation.")
      ->required();

  app.add_option("--marg-data", marg_data_path, "Path to cache folder.")
      ->required();

  app.add_option("--result-path", result_path,
                 "Path to result file where the system will write RMSE ATE.");

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  load_data(cam_calib_path, marg_data_path);

  granite::VioConfig config;

  nrf_mapper.reset(new granite::NfrMapper(calib, config));

  for (auto& kv : marg_data) {
    nrf_mapper->addMargData(kv.second);
  }

  computeEdgeVis();

  std::cout << "roll_pitch_factors.size() " << roll_pitch_factors.size()
            << std::endl;
  std::cout << "rel_pose_factors.size() " << rel_pose_factors.size()
            << std::endl;

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-8.4, -8.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::CreateDisplay()
            .SetAspect(-640 / 480.0)
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetHandler(new pangolin::Handler3D(camera));

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      display3D.Activate(camera);
      glClearColor(1.f, 1.f, 1.f, 1.0f);

      draw_scene();

      pangolin::FinishFrame();

      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  } else {
    setup_points();
    optimize();

    if (!result_path.empty()) {
      double error = alignButton();

      std::ofstream os(result_path);
      os << error << std::endl;
      os.close();
    }
  }

  return 0;
}

void draw_scene() {
  glPointSize(3);
  glColor3f(1.0, 0.0, 0.0);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glColor3ubv(pose_color);
  if (show_points) pangolin::glDrawPoints(mapper_points);

  glColor3ubv(gt_color);
  pangolin::glDrawLineStrip(gt_frame_t_w_i);

  glColor3ubv(cam_color);
  pangolin::glDrawLineStrip(vio_t_w_i);

  glColor3f(0.0, 1.0, 0.0);
  if (show_edges) pangolin::glDrawLines(edges_vis);

  glColor3f(1.0, 0.0, 1.0);
  if (show_edges) pangolin::glDrawLines(roll_pitch_vis);

  glColor3f(1.0, 0.0, 0.0);
  if (show_edges) pangolin::glDrawLines(rel_edges_vis);

  for (const auto& kv : nrf_mapper->getFramePoses()) {
    pangolin::glDrawAxis(kv.second.getPose().matrix(), 0.1);
  }

  pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
}

void load_data(const std::string& calib_path, const std::string& cache_path) {
  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
      std::cout << "Loaded camera with " << calib.intrinsics.size()
                << " cameras" << std::endl;

    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  {
    std::string path = cache_path + "/gt_spline.cereal";
    std::cout << "path " << path << std::endl;

    std::ifstream is(path, std::ios::binary);

    if (is.is_open()) {
      cereal::JSONInputArchive archive(is);

      int64_t t_ns;
      Eigen::aligned_vector<Sophus::SE3d> knots;

      archive(cereal::make_nvp("t_ns", t_ns));
      archive(cereal::make_nvp("knots", knots));

      archive(cereal::make_nvp("gt_observations", gt_observations));
      archive(cereal::make_nvp("noisy_observations", noisy_observations));

      std::cout << "path " << path << std::endl;
      std::cout << "t_ns " << t_ns << std::endl;
      std::cout << "knots " << knots.size() << std::endl;

      gt_spline.reset(new granite::Se3Spline<5>(t_ns));

      for (size_t i = 0; i < knots.size(); i++) {
        gt_spline->knots_push_back(knots[i]);
      }

      is.close();
    } else {
      std::cerr << "could not open " << path << std::endl;
      std::abort();
    }
  }

  {
    int64_t dt_ns = int64_t(1e9) / 50;

    for (int64_t t_ns = 0; t_ns < gt_spline->maxTimeNs(); t_ns += dt_ns) {
      gt_frame_t_w_i.emplace_back(gt_spline->pose(t_ns).translation());
      gt_frame_t_ns.emplace_back(t_ns);
    }
  }

  granite::MargDataLoader mdl;
  tbb::concurrent_bounded_queue<granite::MargData::Ptr> marg_queue;
  mdl.out_marg_queue = &marg_queue;

  mdl.start(cache_path);

  while (true) {
    granite::MargData::Ptr data;
    marg_queue.pop(data);

    if (data.get()) {
      int64_t t_ns = *data->kfs_to_marg.begin();
      marg_data[t_ns] = data;
    } else {
      break;
    }
  }

  std::cout << "Loaded " << marg_data.size() << " marg data." << std::endl;
}

void processMargData(granite::MargData& m) {
  GRANITE_ASSERT(m.aom.total_size == size_t(m.abs_H.cols()));

  std::cout << "rank " << m.abs_H.fullPivLu().rank() << " size "
            << m.abs_H.cols() << std::endl;

  granite::AbsOrderMap aom_new;
  std::set<int> idx_to_keep;
  std::set<int> idx_to_marg;

  for (const auto& kv : m.aom.abs_order_map) {
    if (kv.second.second == se3_SIZE) {
      for (size_t i = 0; i < se3_SIZE; i++)
        idx_to_keep.emplace(kv.second.first + i);
      aom_new.abs_order_map.emplace(kv);
      aom_new.total_size += se3_SIZE;
    } else if (kv.second.second == se3_VEL_BIAS_SIZE) {
      if (m.kfs_all.count(kv.first) > 0) {
        for (size_t i = 0; i < se3_SIZE; i++)
          idx_to_keep.emplace(kv.second.first + i);
        for (size_t i = se3_SIZE; i < se3_VEL_BIAS_SIZE; i++)
          idx_to_marg.emplace(kv.second.first + i);

        aom_new.abs_order_map[kv.first] =
            std::make_pair(aom_new.total_size, se3_SIZE);
        aom_new.total_size += se3_SIZE;

        granite::PoseStateWithLin p = m.frame_states.at(kv.first);
        m.frame_poses[kv.first] = p;
        m.frame_states.erase(kv.first);
      } else {
        for (size_t i = 0; i < se3_VEL_BIAS_SIZE; i++)
          idx_to_marg.emplace(kv.second.first + i);
        m.frame_states.erase(kv.first);
      }
    } else {
      std::cerr << "Unknown size" << std::endl;
      std::abort();
    }

    std::cout << kv.first << " " << kv.second.first << " " << kv.second.second
              << std::endl;
  }

  Eigen::MatrixXd marg_H_new;
  Eigen::VectorXd marg_b_new;
  granite::KeypointVioEstimator::marginalizeHelper(
      m.abs_H, m.abs_b, idx_to_keep, idx_to_marg, marg_H_new, marg_b_new);

  std::cout << "new rank " << marg_H_new.fullPivLu().rank() << " size "
            << marg_H_new.cols() << std::endl;

  m.abs_H = marg_H_new;
  m.abs_b = marg_b_new;
  m.aom = aom_new;

  GRANITE_ASSERT(m.aom.total_size == size_t(m.abs_H.cols()));
}

void extractNonlinearFactors(granite::MargData& m) {
  size_t asize = m.aom.total_size;
  std::cout << "asize " << asize << std::endl;

  Eigen::MatrixXd cov_old;
  cov_old.setIdentity(asize, asize);
  m.abs_H.ldlt().solveInPlace(cov_old);

  int64_t kf_id = *m.kfs_to_marg.cbegin();
  int kf_start_idx = m.aom.abs_order_map.at(kf_id).first;

  auto state_kf = m.frame_poses.at(kf_id);

  Sophus::SE3d T_w_i_kf = state_kf.getPose();

  Eigen::Vector3d pos = T_w_i_kf.translation();
  Eigen::Vector3d yaw_dir_body =
      T_w_i_kf.so3().inverse() * Eigen::Vector3d::UnitX();

  Sophus::Matrix<double, 3, se3_SIZE> d_pos_d_T_w_i;
  Sophus::Matrix<double, 1, se3_SIZE> d_yaw_d_T_w_i;
  Sophus::Matrix<double, 2, se3_SIZE> d_rp_d_T_w_i;

  granite::absPositionError(T_w_i_kf, pos, &d_pos_d_T_w_i);
  granite::yawError(T_w_i_kf, yaw_dir_body, &d_yaw_d_T_w_i);
  granite::rollPitchError(T_w_i_kf, T_w_i_kf.so3(), &d_rp_d_T_w_i);

  {
    Eigen::MatrixXd J;
    J.setZero(se3_SIZE, asize);
    J.block<3, se3_SIZE>(0, kf_start_idx) = d_pos_d_T_w_i;
    J.block<1, se3_SIZE>(3, kf_start_idx) = d_yaw_d_T_w_i;
    J.block<2, se3_SIZE>(4, kf_start_idx) = d_rp_d_T_w_i;

    Sophus::Matrix6d cov_new = J * cov_old * J.transpose();

    // std::cout << "cov_new\n" << cov_new << std::endl;

    granite::RollPitchFactor rpf;
    rpf.t_ns = kf_id;
    rpf.R_w_i_meas = T_w_i_kf.so3();
    rpf.cov_inv = cov_new.block<2, 2>(4, 4).inverse();

    roll_pitch_factors.emplace_back(rpf);
  }

  for (int64_t other_id : m.kfs_all) {
    if (m.frame_poses.count(other_id) == 0 || other_id == kf_id) {
      continue;
    }

    auto state_o = m.frame_poses.at(other_id);

    Sophus::SE3d T_w_i_o = state_o.getPose();
    Sophus::SE3d T_kf_o = T_w_i_kf.inverse() * T_w_i_o;

    int o_start_idx = m.aom.abs_order_map.at(other_id).first;

    Sophus::Matrix6d d_res_d_T_w_i, d_res_d_T_w_j;
    granite::relPoseError(T_kf_o, T_w_i_kf, T_w_i_o, &d_res_d_T_w_i,
                          &d_res_d_T_w_j);

    Eigen::MatrixXd J;
    J.setZero(se3_SIZE, asize);
    J.block<se3_SIZE, se3_SIZE>(0, kf_start_idx) = d_res_d_T_w_i;
    J.block<se3_SIZE, se3_SIZE>(0, o_start_idx) = d_res_d_T_w_j;

    Sophus::Matrix6d cov_new = J * cov_old * J.transpose();
    granite::RelPoseFactor rpf;
    rpf.t_i_ns = kf_id;
    rpf.t_j_ns = other_id;
    rpf.T_i_j = T_kf_o;
    rpf.cov_inv.setIdentity();
    cov_new.ldlt().solveInPlace(rpf.cov_inv);

    // std::cout << "rpf.cov_inv\n" << rpf.cov_inv << std::endl;

    rel_pose_factors.emplace_back(rpf);
  }
}

void computeEdgeVis() {
  edges_vis.clear();
  for (const auto& kv1 : nrf_mapper->lmdb.getObservations()) {
    for (const auto& kv2 : kv1.second) {
      Eigen::Vector3d p1 = nrf_mapper->getFramePoses()
                               .at(kv1.first.frame_id)
                               .getPose()
                               .translation();
      Eigen::Vector3d p2 = nrf_mapper->getFramePoses()
                               .at(kv2.first.frame_id)
                               .getPose()
                               .translation();

      edges_vis.emplace_back(p1);
      edges_vis.emplace_back(p2);
    }
  }

  roll_pitch_vis.clear();
  for (const auto& v : nrf_mapper->roll_pitch_factors) {
    const Sophus::SE3d& T_w_i =
        nrf_mapper->getFramePoses().at(v.t_ns).getPose();

    Eigen::Vector3d p = T_w_i.translation();
    Eigen::Vector3d d =
        v.R_w_i_meas * T_w_i.so3().inverse() * (-Eigen::Vector3d::UnitZ());

    roll_pitch_vis.emplace_back(p);
    roll_pitch_vis.emplace_back(p + 0.1 * d);
  }

  rel_edges_vis.clear();
  for (const auto& v : nrf_mapper->rel_pose_factors) {
    Eigen::Vector3d p1 =
        nrf_mapper->getFramePoses().at(v.t_i_ns).getPose().translation();
    Eigen::Vector3d p2 =
        nrf_mapper->getFramePoses().at(v.t_j_ns).getPose().translation();

    rel_edges_vis.emplace_back(p1);
    rel_edges_vis.emplace_back(p2);
  }
}

void optimize() {
  nrf_mapper->optimize();
  nrf_mapper->get_current_points(mapper_points, mapper_point_ids);
  //  nrf_mapper->get_current_points_with_color(mapper_points,
  //  mapper_points_color,
  //                                            mapper_point_ids);

  computeEdgeVis();
}

void randomInc() {
  Sophus::Vector6d rnd = Sophus::Vector6d::Random().array().abs();
  Sophus::SE3d random_inc = Sophus::se3_expd(rnd / 10);

  for (auto& kv : nrf_mapper->getFramePoses()) {
    Sophus::SE3d pose = random_inc * kv.second.getPose();
    granite::PoseStateWithLin p(kv.first, pose);
    kv.second = p;
  }

  computeEdgeVis();
}

void randomYawInc() {
  Sophus::Vector6d rnd;
  rnd.setZero();
  rnd[5] = std::abs(Eigen::Vector2d::Random()[0]);

  Sophus::SE3d random_inc = Sophus::se3_expd(rnd);

  std::cout << "random_inc\n" << random_inc.matrix() << std::endl;

  for (auto& kv : nrf_mapper->getFramePoses()) {
    Sophus::SE3d pose = random_inc * kv.second.getPose();
    granite::PoseStateWithLin p(kv.first, pose);
    kv.second = p;
  }

  computeEdgeVis();
}

double alignButton() {
  Eigen::aligned_vector<Eigen::Vector3d> filter_t_w_i;
  std::vector<int64_t> filter_t_ns;

  for (const auto& kv : nrf_mapper->getFramePoses()) {
    filter_t_ns.emplace_back(kv.first);
    filter_t_w_i.emplace_back(kv.second.getPose().translation());
  }

  Sophus::SE3d T_gt_est;
  Sophus::Sim3d sT_gt_est;

  return granite::alignSVD(filter_t_ns, filter_t_w_i, gt_frame_t_ns,
                           gt_frame_t_w_i, T_gt_est, sT_gt_est).first;
}

void setup_points() {
  for (auto& kv : nrf_mapper->getFramePoses()) {
    for (size_t i = 0; i < calib.intrinsics.size(); i++) {
      granite::TimeCamId tcid(kv.first, i);
      auto obs = noisy_observations.at(tcid);

      granite::KeypointsData kd;
      kd.corners = obs.pos;

      nrf_mapper->feature_corners[tcid] = kd;

      for (size_t j = 0; j < kd.corners.size(); j++) {
        nrf_mapper->feature_tracks[obs.id[j]][tcid] = j;
      }
    }
  }

  for (auto it = nrf_mapper->feature_tracks.cbegin();
       it != nrf_mapper->feature_tracks.cend();) {
    if (it->second.size() < 5) {
      it = nrf_mapper->feature_tracks.erase(it);
    } else {
      ++it;
    }
  }

  std::cerr << "nrf_mapper->feature_tracks.size() "
            << nrf_mapper->feature_tracks.size() << std::endl;

  nrf_mapper->setup_opt();

  nrf_mapper->get_current_points(mapper_points, mapper_point_ids);
}
