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

#include <sophus/se3.hpp>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>

#include <granite/io/dataset_io.h>
#include <granite/io/marg_data_io.h>
#include <granite/spline/se3_spline.h>
#include <granite/utils/common_types.h>
#include <granite/utils/exceptions.h>
#include <granite/vi_estimator/vio_estimator.h>
#include <granite/calibration/calibration.hpp>

#include <granite/serialization/headers_serialization.h>

#include <granite/utils/vis_utils.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <thread>

class VioDataset {
 public:
  VioDataset(const std::string& cam_calib_path, const std::string& config_path,
             const std::string& dataset_type, const std::string& dataset_path,
             const std::string& marg_data_path, const std::string& result_path,
             const std::string& stats_path, const std::string& trajectory_fmt,
             const bool show_gui, const bool use_imu, const bool step_by_step,
             const bool print_queue)
      : show_gui(show_gui),
        step_by_step(step_by_step),
        show_frame("ui.show_frame", 0, 0, 1500),

        show_flow("ui.show_flow", false, false, true),
        show_obs("ui.show_obs", true, false, true),
        show_ids("ui.show_ids", false, false, true),
        show_epipolar("ui.show_epipolar", false, false, true),

        show_kf("plot_panel.show_kf", true, false, true),
        show_est_pos("plot_panel.show_est_pos", true, false, true),
        show_est_vel("plot_panel.show_est_vel", false, false, true),
        show_est_bg("plot_panel.show_est_bg", false, false, true),
        show_est_ba("plot_panel.show_est_ba", false, false, true),

        show_gt_traj("scene_panel.show_gt_traj", true, false, true),
        show_gt_pos("plot_panel.show_gt_pos", false, true),
        show_gt_vel("plot_panel.show_gt_vel", false, true),
        show_scale("plot_panel.show_scale", false, true),
        show_entropy("plot_panel.show_entropy", false, true),

        continue_btn("ui.continue", false, false, true),
        continue_fast("ui.continue_fast", true, false, true),

        euroc_fmt("ui.euroc_fmt", true, false, true),
        tum_rgbd_fmt("ui.tum_rgbd_fmt", false, false, true),
        kitti_fmt("ui.kitti_fmt", false, false, true),

        follow("scene_panel.follow", true, false, true),
        print_queue(print_queue),
        trajectory_fmt(trajectory_fmt),
        result_path(result_path),
        stats_path(stats_path),
        marg_data_path(marg_data_path),
        use_imu(use_imu) {
    imu_data_queue.set_capacity(300);

    // load configuration
    if (!config_path.empty()) {
      vio_config.load(config_path);

      if (vio_config.vio_enforce_realtime) {
        vio_config.vio_enforce_realtime = false;
        std::cout
            << "The option vio_config.vio_enforce_realtime was enabled, "
               "but it should only be used with the live executables (supply "
               "images at a constant framerate). This executable runs on the "
               "datasets and processes images as fast as it can, so the option "
               "will be disabled. "
            << std::endl;
      }
    }

    loadCamCalib(cam_calib_path);

    // load dataset
    {
      granite::DatasetIoInterfacePtr dataset_io =
          granite::DatasetIoFactory::getDatasetIo(dataset_type);

      bool success = dataset_io->read(dataset_path, calib.intrinsics.size());

      if (!success) {
        std::abort();
      }

      vio_dataset = dataset_io->get_data();
    }

    if (calib.T_i_c.size() == 1 && use_imu) {
      std::cerr << "Monocular-inertial is not yet supported. Set --use-imu to 0!" << std::endl;
      throw granite::NotImplementedException();
    }

    // populate ground truth data structures
    start_t_ns = vio_dataset->get_image_timestamps().front();

    show_frame.Meta().range[1] = vio_dataset->get_image_timestamps().size() - 1;
    show_frame.Meta().gui_changed = true;

    for (size_t i = 0; i < vio_dataset->get_gt_pose_data().size(); i++) {
      gt_t_ns.push_back(vio_dataset->get_gt_timestamps()[i]);

      const Sophus::SE3d T_w_i = vio_dataset->get_gt_pose_data()[i];
      const Sophus::SE3d T_w_c0 = T_w_i * calib.T_i_c.at(0);
      gt_T_w_i.push_back(T_w_i);
      gt_t_w_i.push_back(T_w_i.translation());
      gt_T_w_c0.push_back(T_w_c0);
      gt_t_w_c0.push_back(T_w_c0.translation());
    }

    T_gt_est.emplace_back(
        !vio_dataset->get_gt_pose_data().empty()
            ? interpolate_trajectory(vio_dataset->get_gt_timestamps(),
                                     vio_dataset->get_gt_pose_data(),
                                     vio_dataset->get_image_timestamps().at(0))
            : Sophus::SE3d());
    sT_gt_est.emplace_back(
        Sophus::Sim3d(Sophus::RxSO3d(T_gt_est.at(0).unit_quaternion()),
                      T_gt_est.at(0).translation()));

    populate_gt_data_log();

    // initialize system

    // initialize optical flow
    opt_flow_ptr =
        granite::OpticalFlowFactory::getOpticalFlow(vio_config, calib);

    // initialize vio

    vio = granite::VioEstimatorFactory::getVioEstimator(
        vio_config, calib, granite::constants::g, use_imu);

    vio->vision_data_queue = &opt_flow_ptr->output_queue;
    vio->imu_data_queue = &imu_data_queue;

    if (show_gui) vio->out_vis_queue = &out_vis_queue;
    vio->out_state_queue = &out_state_queue;
  }

  void run() {
    auto time_start = std::chrono::high_resolution_clock::now();

    if (!marg_data_path.empty()) {
      marg_data_saver.reset(new granite::MargDataSaver(marg_data_path));
      vio->out_marg_queue = &marg_data_saver->in_marg_queue;

      // Save gt.
      {
        std::string p = marg_data_path + "/gt.cereal";
        std::ofstream os(p, std::ios::binary);

        {
          cereal::BinaryOutputArchive archive(os);
          archive(gt_t_ns);
          archive(gt_t_w_i);
        }
        os.close();
      }
    }

    vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    feedImagesLoop();
    if (use_imu) feedImuLoop();
    if (show_gui) visDataLoop();
    stateLoop();
    queueSizePrinterLoop();

    if (show_gui) {
      initGui();
      guiLoop();

      // if we reached this point the user commanded a shutdown

      // force shutdown
      terminate = true;
      cv.notify_all();
      opt_flow_ptr->quit(true);
      vio->quit();
      if (marg_data_saver) marg_data_saver->quit();
      imu_data_queue.abort();
      imu_data_queue.clear();
      out_vis_queue.abort();
      out_vis_queue.clear();
      out_state_queue.abort();
      out_state_queue.clear();
    }

    // wait for clean up
    if (vis_data_receiver && vis_data_receiver->joinable())
      vis_data_receiver->join();
    if (state_receiver && state_receiver->joinable()) state_receiver->join();
    if (marg_data_saver) marg_data_saver->join();
    opt_flow_ptr->join();
    vio->join();
    if (image_feeder && image_feeder->joinable()) image_feeder->join();
    if (imu_feeder && imu_feeder->joinable()) imu_feeder->join();
    terminate = true;
    if (queue_size_printer && queue_size_printer->joinable())
      queue_size_printer->join();

    // do evaluation and write results and stats to disk
    auto time_end = std::chrono::high_resolution_clock::now();
    auto exec_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        time_end - time_start);

    if (!trajectory_fmt.empty()) {
      if (trajectory_fmt == "kitti") {
        kitti_fmt = true;
        euroc_fmt = false;
        tum_rgbd_fmt = false;
      }
      if (trajectory_fmt == "euroc") {
        euroc_fmt = true;
        kitti_fmt = false;
        tum_rgbd_fmt = false;
      }
      if (trajectory_fmt == "tum") {
        tum_rgbd_fmt = true;
        euroc_fmt = false;
        kitti_fmt = false;
      }

      saveTrajectoryButton();
    }

    std::pair<double, double> errors;
    size_t best_idx = 0;
    {
      size_t best_length = 0;
      for (size_t idx = 0; idx < vio_t_ns.size(); idx++) {
        if (vio_t_ns.at(idx).size() > best_length) {
          best_idx = idx;
          best_length = vio_t_ns.at(idx).size();
        }
      }
    }

    std::vector<granite::FrameId> best_t_ns(vio_t_ns.at(best_idx).begin(),
                                           vio_t_ns.at(best_idx).end());

    if (calib.intrinsics.size() > 1 || use_imu) {
      Eigen::aligned_vector<Eigen::Vector3d> best_t_w_i(
          vio_t_w_i.at(best_idx).begin(), vio_t_w_i.at(best_idx).end());

      errors = granite::alignSVD(best_t_ns, best_t_w_i, gt_t_ns, gt_t_w_i,
                                T_gt_est.at(best_idx), sT_gt_est.at(best_idx));
    } else {
      Eigen::aligned_vector<Eigen::Vector3d> best_t_w_c0(
          vio_t_w_c0.at(best_idx).begin(), vio_t_w_c0.at(best_idx).end());

      errors = granite::alignSVD(best_t_ns, best_t_w_c0, gt_t_ns, gt_t_w_c0,
                                T_gt_est.at(best_idx), sT_gt_est.at(best_idx));
    }

    if (!result_path.empty()) {
      std::ofstream os(result_path);
      {
        cereal::JSONOutputArchive ar(os);
        ar(cereal::make_nvp("se3_rms_ate", errors.first));
        ar(cereal::make_nvp("sim3_rms_ate", errors.second));
        ar(cereal::make_nvp("num_frames",
                            vio_dataset->get_image_timestamps().size()));
        ar(cereal::make_nvp("exec_time_ns", exec_time_ns.count()));
        ar(cereal::make_nvp("num_maps", vio_t_ns.size()));
      }
      os.close();
    }

    if (show_gui && !stats_path.empty()) {
      write_stats(stats_path);
    }
  }

  /// @brief Constructs the visualization Graphical User Interface (GUI)
  void initGui() {
    pangolin::CreateWindowAndBind("Basalt Dataset", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View& main_view = pangolin::Display("main")
                                    .SetBounds(0.0, 1.0, 0.0, 1.0)
                                    .SetLayout(pangolin::LayoutEqualVertical);

    pangolin::View& top_view = pangolin::Display("top");
    main_view.AddDisplay(top_view);

    pangolin::View& main_panel = pangolin::CreatePanel("ui").SetBounds(
        0.0, 1.0, 0.0, pangolin::Attach::Pix(PANEL_WIDTH));
    top_view.AddDisplay(main_panel);

    // images

    img_view_display =
        &pangolin::CreateDisplay()
             .SetBounds(0.0, 1.0, pangolin::Attach::Pix(PANEL_WIDTH),
                        pangolin::Attach::Pix(PANEL_WIDTH + 450))
             .SetLayout(pangolin::LayoutEqualVertical);
    top_view.AddDisplay(*img_view_display);

    while (img_view.size() < calib.intrinsics.size()) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display->AddDisplay(*iv);
      iv->extern_draw_function = [this, idx](pangolin::View& v) {
        draw_image_overlay(v, idx);
      };
    }

    // 3D scene

    pangolin::View& scene_view =
        pangolin::Display("scene_wrapper")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(PANEL_WIDTH + 450), 1.0);

    top_view.AddDisplay(scene_view);

    pangolin::View& scene_panel =
        pangolin::CreatePanel("scene_panel")
            .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(PANEL_WIDTH));
    scene_view.AddDisplay(scene_panel);

    // Eigen::Vector3d cam_p(-0.5, -3, -5);
    Eigen::Vector3d cam_p =
        (T_gt_est.at(0) * vio->getT_w_i_init() * calib.T_i_c[0]).translation();
    Eigen::Vector3d viewer_p_c0(0, -3, -5);
    Eigen::Vector3d viewer_p =
        T_gt_est.at(0) * vio->getT_w_i_init() * calib.T_i_c[0] * viewer_p_c0;

    camera = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(viewer_p[0], viewer_p[1], viewer_p[2],
                                  cam_p[0], cam_p[1], cam_p[2],
                                  pangolin::AxisZ));

    display3D =
        &pangolin::CreateDisplay()
             .SetAspect(-640 / 480.0)
             .SetBounds(0.0, 1.0, pangolin::Attach::Pix(PANEL_WIDTH), 1.0)
             .SetHandler(new pangolin::Handler3D(camera, pangolin::AxisZ));

    display3D->extern_draw_function = [&](pangolin::View& v) { draw_scene(v); };

    scene_view.AddDisplay(*display3D);

    // plot

    pangolin::View& bottom_view = pangolin::Display("bottom");
    main_view.AddDisplay(bottom_view);

    pangolin::View& plot_panel =
        pangolin::CreatePanel("plot_panel")
            .SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(PANEL_WIDTH));
    bottom_view.AddDisplay(plot_panel);

    pangolin::View& plot_display = pangolin::CreateDisplay().SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(PANEL_WIDTH), 1.0);

    plotter = new pangolin::Plotter(&imu_data_log, 0.0, 100, -10.0, 10.0, 0.01f,
                                    0.001f);
    plot_display.AddDisplay(*plotter);
    bottom_view.AddDisplay(plot_display);

    using Button = pangolin::Var<std::function<void(void)>>;

    Button next_step_btn("ui.next_step",
                         std::bind(&VioDataset::next_step, this));
    Button prev_step_btn("ui.prev_step",
                         std::bind(&VioDataset::prev_step, this));
    Button align_btn("ui.align", std::bind(&VioDataset::alignButton, this));
    Button save_traj_btn("ui.save_traj",
                         std::bind(&VioDataset::saveTrajectoryButton, this));
    pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT,
        std::bind(&VioDataset::next_step, this));
    pangolin::RegisterKeyPressCallback(
        pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_LEFT,
        std::bind(&VioDataset::prev_step, this));
  }

  /// @brief Runs until user termination. Updates the GUI
  void guiLoop() {
    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      // let the view follow the camera
      if (follow) {
        size_t frame_id = show_frame;
        int64_t t_ns = vio_dataset->get_image_timestamps()[frame_id];

        granite::VioVisualizationData::Ptr vis_data;
        for (const auto& vm : vis_map) {
          auto it = vm.find(t_ns);

          if (it != vm.end()) {
            vis_data = it->second;
            break;
          }
        }

        if (vis_data) {
          Sophus::SE3d T_w_i;
          if (!vis_data->states.empty()) {
            T_w_i = vis_data->states.back();
          } else if (!vis_data->frames.empty()) {
            T_w_i = vis_data->frames.back();
          }

          T_w_i = calib.intrinsics.size() > 1
                      ? T_gt_est.at(vis_data->map_idx) * T_w_i
                      : sT_gt_est.at(vis_data->map_idx) * T_w_i;

          // T_w_i.so3() = Sophus::SO3d();

          camera.Follow(T_w_i.matrix());
        }
      }

      display3D->Activate(camera);
      glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

      img_view_display->Activate();

      // the timestamp for which the system state shall be displayed changed
      if (show_frame.GuiChanged()) {
        for (size_t cam_id = 0; cam_id < calib.intrinsics.size(); cam_id++) {
          size_t frame_id = static_cast<size_t>(show_frame);
          int64_t timestamp = vio_dataset->get_image_timestamps()[frame_id];

          std::vector<granite::ImageData> img_vec =
              vio_dataset->get_image_data(timestamp);

          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          if (std::is_same_v<granite::PixelType, uint8_t>) {
            fmt.gltype = GL_UNSIGNED_BYTE;
            fmt.scalable_internal_format = GL_LUMINANCE8;
          } else if (std::is_same_v<granite::PixelType, uint16_t>) {
            fmt.gltype = GL_UNSIGNED_SHORT;
            fmt.scalable_internal_format = GL_LUMINANCE16;
          } else {
            throw granite::NotImplementedException();
          }

          if (img_vec[cam_id].img.get())
            img_view[cam_id]->SetImage(
                img_vec[cam_id].img->ptr, img_vec[cam_id].img->w,
                img_vec[cam_id].img->h, img_vec[cam_id].img->pitch, fmt);
        }

        draw_plots();
      }

      // the plot selection changed
      if (show_est_vel.GuiChanged() || show_est_pos.GuiChanged() ||
          show_est_ba.GuiChanged() || show_est_bg.GuiChanged() ||
          show_gt_pos.GuiChanged() || show_gt_vel.GuiChanged() ||
          show_scale.GuiChanged() || show_entropy.GuiChanged() ||
          show_kf.GuiChanged()) {
        draw_plots();
      }

      // the trajectory output format changed
      if (euroc_fmt.GuiChanged()) {
        euroc_fmt = true;
        tum_rgbd_fmt = false;
        kitti_fmt = false;
      }

      if (tum_rgbd_fmt.GuiChanged()) {
        tum_rgbd_fmt = true;
        euroc_fmt = false;
        kitti_fmt = false;
      }

      if (kitti_fmt.GuiChanged()) {
        kitti_fmt = true;
        euroc_fmt = false;
        tum_rgbd_fmt = false;
      }

      //      if (record) {
      //        main_display.RecordOnRender(
      //            "ffmpeg:[fps=50,bps=80000000,unique_filename]///tmp/"
      //            "vio_screencap.avi");
      //        record = false;
      //      }

      pangolin::FinishFrame();

      // continue computation
      if (continue_btn ||
          last_pushed_t_ns <
              vio_dataset->get_image_timestamps().at(show_frame)) {
        if (!next_step()) {
          // end of dataset was reached
          continue_btn = false;
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      } else {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }

      if (continue_fast) {
        // display last computed state
        granite::FrameId t_ns = vio->getLastProcessedt_ns();
        if (timestamp_to_id.count(t_ns)) {
          show_frame = timestamp_to_id[t_ns];
          show_frame.Meta().gui_changed = true;
        }

        if (vio->isFinished()) {
          continue_fast = false;
        }
      }
    }
  }

  // Feed functions

  /// @brief Starts thread that loads images from disk and publishes them to the
  /// input queue of the optical flow computer. Runs until completion or
  /// terminate is true. Indicates the end of the dataset with a nullptr. If
  /// step_by_step is used it waits until cv is notified (e.g. user pushed
  /// next_step button).
  void feedImagesLoop() {
    image_feeder.reset(new std::thread([&]() {
      std::cout << "Started image feeder thread" << std::endl;

      for (size_t i = 0;
           !terminate && i < vio_dataset->get_image_timestamps().size(); i++) {
        if (step_by_step) {
          std::unique_lock<std::mutex> lk(m);
          cv.wait(lk);
        }
        granite::OpticalFlowInput::Ptr data(new granite::OpticalFlowInput);

        data->t_ns = vio_dataset->get_image_timestamps()[i];
        data->img_data = vio_dataset->get_image_data(data->t_ns);

        timestamp_to_id[data->t_ns] = i;

        try {
          opt_flow_ptr->input_queue.push(data);
        } catch (const tbb::user_abort&) {
          break;
        };
        last_pushed_t_ns = data->t_ns;
      }

      if (!terminate) {
        // Indicate the end of the sequence
        try {
          opt_flow_ptr->input_queue.push(nullptr);
        } catch (const tbb::user_abort&) {
        };
      }

      std::cout << "Finished image feeder thread" << std::endl;
    }));
  }

  /// @brief Starts thread that pushes imu data from dataset into imu_data_queue
  /// until completion or terminate is true. Indicates the end of the dataset
  /// with a nullptr.
  void feedImuLoop() {
    imu_feeder.reset(new std::thread([&]() {
      std::cout << "Started IMU feeder thread" << std::endl;

      for (size_t i = 0; !terminate && i < vio_dataset->get_gyro_data().size();
           i++) {
        granite::ImuData::Ptr data(new granite::ImuData);
        data->t_ns = vio_dataset->get_gyro_data()[i].timestamp_ns;

        data->accel = vio_dataset->get_accel_data()[i].data;
        data->gyro = vio_dataset->get_gyro_data()[i].data;

        try {
          imu_data_queue.push(data);
        } catch (const tbb::user_abort&) {
          break;
        };
      }
      if (!terminate) {
        // Indicate the end of the sequence
        try {
          imu_data_queue.push(nullptr);
        } catch (const tbb::user_abort&) {
        };
      }
      std::cout << "Finished IMU feeder thread" << std::endl;
    }));
  }

  // Receive functions

  /// @brief Starts thread that listens to out_vis_queue and stores data in
  /// vis_map and other visualization data structures
  void visDataLoop() {
    vis_data_receiver.reset(new std::thread([&]() {
      std::cout << "Started visualization receiver thread" << std::endl;

      granite::VioVisualizationData::Ptr data;

      while (!terminate) {
        try {
          out_vis_queue.pop(data);
        } catch (const tbb::user_abort&) {
          break;
        };

        if (!data.get()) break;

        while (vis_map.size() <= data->map_idx) {
          vis_map.emplace_back();
        }

        vis_map[data->map_idx][data->t_ns] = data;

        vio_kf.emplace_back(data->take_kf);

        std::vector<float> vals;
        vals.push_back((data->t_ns - start_t_ns) * 1e-9);

        // compute Sim3 alignment scale of current optimization window
        double scale = std::numeric_limits<double>::quiet_NaN();
        if (data->states.size() + data->frames.size() >= 3) {
          std::vector<int64_t> window_t_ns;
          window_t_ns.insert(std::end(window_t_ns),
                             std::begin(data->states_t_ns),
                             std::end(data->states_t_ns));
          window_t_ns.insert(std::end(window_t_ns),
                             std::begin(data->frames_t_ns),
                             std::end(data->frames_t_ns)
                             /*std::next(std::begin(data->frames_t_ns), 3)*/);
          Eigen::aligned_vector<Eigen::Vector3d> window_t_w_i;
          std::transform(data->states.begin(), data->states.end(),
                         std::back_inserter(window_t_w_i),
                         [](Sophus::SE3d& T) -> Eigen::Vector3d {
                           return T.translation();
                         });
          std::transform(
              data->frames.begin(),
              data->frames.end() /* std::next(data->frames.begin(), 3)*/,
              std::back_inserter(window_t_w_i),
              [](Sophus::SE3d& T) -> Eigen::Vector3d {
                return T.translation();
              });
          Eigen::aligned_vector<Eigen::Vector3d> window_t_w_c0;
          std::transform(data->states.begin(), data->states.end(),
                         std::back_inserter(window_t_w_c0),
                         [&](Sophus::SE3d& T) -> Eigen::Vector3d {
                           return (T * calib.T_i_c.at(0)).translation();
                         });
          std::transform(
              data->frames.begin(),
              data->frames.end() /*std::next(data->frames.begin(), 3)*/,
              std::back_inserter(window_t_w_c0),
              [&](Sophus::SE3d& T) -> Eigen::Vector3d {
                return (T * calib.T_i_c.at(0)).translation();
              });

          Sophus::SE3d window_T_gt_est;
          Sophus::Sim3d window_sT_gt_est;

          if (calib.intrinsics.size() > 1 || use_imu) {
            granite::alignSVD(window_t_ns, window_t_w_i, gt_t_ns, gt_t_w_i,
                             window_T_gt_est, window_sT_gt_est, false);
          } else {
            granite::alignSVD(window_t_ns, window_t_w_c0, gt_t_ns, gt_t_w_c0,
                             window_T_gt_est, window_sT_gt_est, false);
          }
          scale = 1.0 / window_sT_gt_est.scale();
        }

        vals.push_back(scale);
        vals.push_back(data->negative_entropy_last_frame);
        vals.push_back(data->average_negative_entropy_last_frame);

        vis_data_log.Log(vals);

        scale_vec.push_back(scale);
        n_entropy_last_frame_vec.push_back(data->negative_entropy_last_frame);
        avg_n_entropy_last_frame_vec.push_back(
            data->average_negative_entropy_last_frame);
      }

      std::cout << "Finished visualization receiver thread" << std::endl;
    }));
  }

  /// @brief Starts thread that listens to out_state_queue and populates some
  /// data structures for evaluation
  void stateLoop() {
    state_receiver.reset(new std::thread([&]() {
      std::cout << "Started state receiver thread" << std::endl;

      granite::VioStateData::Ptr data;

      while (!terminate) {
        try {
          out_state_queue.pop(data);
        } catch (const tbb::user_abort&) {
          break;
        };

        if (!data.get()) break;

        int64_t t_ns = data->state.t_ns;

        // std::cerr << "t_ns " << t_ns << std::endl;
        const Sophus::SE3d T_w_i = data->state.T_w_i;
        const Eigen::Vector3d vel_w_i = data->state.vel_w_i;
        const Eigen::Vector3d bg = data->state.bias_gyro;
        const Eigen::Vector3d ba = data->state.bias_accel;

        while (T_gt_est.size() <= data->map_idx) {
          T_gt_est.emplace_back(T_gt_est.back());
          sT_gt_est.emplace_back(sT_gt_est.back());
        }

        while (vio_t_ns.size() <= data->map_idx) {
          vio_t_w_i.emplace_back();
          vio_t_w_c0.emplace_back();
          vio_T_w_i.emplace_back();
          vio_T_w_c0.emplace_back();
          vio_vel_w_i.emplace_back();
          vio_bg.emplace_back();
          vio_ba.emplace_back();
          vio_t_ns.emplace_back();
        }

        vio_t_w_i[data->map_idx].emplace_back(T_w_i.translation());
        vio_t_w_c0[data->map_idx].emplace_back(
            (T_w_i * calib.T_i_c.at(0)).translation());
        vio_T_w_i[data->map_idx].emplace_back(T_w_i);
        vio_T_w_c0[data->map_idx].emplace_back(T_w_i * calib.T_i_c.at(0));
        vio_vel_w_i[data->map_idx].emplace_back(vel_w_i);
        vio_bg[data->map_idx].emplace_back(bg);
        vio_ba[data->map_idx].emplace_back(ba);
        vio_t_ns[data->map_idx].emplace_back(data->state.t_ns);
        scale_variance.emplace_back(data->scale_variance);
        drift_variance.emplace_back(data->drift_variance);

        if (show_gui) {
          std::vector<float> vals;
          vals.push_back((t_ns - start_t_ns) * 1e-9);

          Eigen::Vector3d v = calib.intrinsics.size() > 1
                                  ? T_gt_est.at(data->map_idx) * vel_w_i
                                  : sT_gt_est.at(data->map_idx) * vel_w_i;

          Eigen::Vector3d t = (calib.intrinsics.size() > 1
                                   ? T_gt_est.at(data->map_idx) * T_w_i
                                   : sT_gt_est.at(data->map_idx) * T_w_i)
                                  .translation();

          for (int i = 0; i < 3; i++) vals.push_back(v[i]);
          for (int i = 0; i < 3; i++) vals.push_back(t[i]);
          for (int i = 0; i < 3; i++) vals.push_back(bg[i]);
          for (int i = 0; i < 3; i++) vals.push_back(ba[i]);

          vio_data_log.Log(vals);

          std::vector<float> vals2;
          vals2.push_back((t_ns - start_t_ns) * 1e-9);
          vals2.push_back(data->scale_variance);
          vals2.push_back(data->drift_variance);
          state_data_log.Log(vals2);
        }
      }

      std::cout << "Finished state receiver thread" << std::endl;
    }));
  }

  /// @brief Starts thread that prints the queue sizes every second. That way
  /// the computationally bottleneck can be found.
  void queueSizePrinterLoop() {
    if (print_queue) {
      queue_size_printer.reset(new std::thread([&]() {
        while (!terminate) {
          std::cout << "opt_flow_ptr->input_queue "
                    << opt_flow_ptr->input_queue.size()
                    << " opt_flow_ptr->output_queue "
                    << opt_flow_ptr->output_queue.size() << " out_state_queue "
                    << out_state_queue.size() << std::endl;
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }
      }));
    }
  }

  /// @brief Updates image display to currently `show_frame`. Draws overlays for
  /// optical flow, landmark reprojection, epipolar curves, ... . Gets data from
  /// vis_map.
  void draw_image_overlay(pangolin::View& v, granite::CamId cam_id) {
    UNUSED(v);

    //  size_t frame_id = show_frame;
    //  granite::TimeCamId tcid =
    //      std::make_pair(vio_dataset->get_image_timestamps()[frame_id],
    //      cam_id);

    glColor3f(1.0, 0.0, 0.0);
    pangolin::GlFont::I()
        .Text("FrameId: %lld", vio_dataset->get_image_timestamps()[show_frame])
        .Draw(5, 25);

    size_t frame_id = show_frame;
    granite::VioVisualizationData::Ptr vis_data;
    for (const auto& vm : vis_map) {
      auto it = vm.find(vio_dataset->get_image_timestamps()[frame_id]);

      if (it != vm.end()) {
        vis_data = it->second;
        break;
      }
    }

    if (show_obs) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (vis_data && cam_id < vis_data->projections.size()) {
        const auto& points = vis_data->projections[cam_id];

        if (points.size() > 0) {
          double min_id = std::numeric_limits<double>::infinity(), max_id = 0.0;

          for (const auto& points2 : vis_data->projections)
            for (const auto& p : points2) {
              if (p[2] > 0.0) {
                min_id = std::min(min_id, p[2]);
                max_id = std::max(max_id, p[2]);
              }
            }

          for (const auto& c : points) {
            const auto& pyr_levels =
                vis_data->opt_flow_res->pyramid_levels.at(cam_id);

            const float radius = 6.5 * (pyr_levels.at(size_t(c[3])) + 1);
            if (c[2] > 0.0) {
              float r, g, b;
              getcolor(c[2] - min_id, max_id - min_id, b, g, r);
              glColor3f(r, g, b);

              pangolin::glDrawCirclePerimeter(c[0], c[1], radius);
            } else {
              glColor3f(0.0, 0.0, 1.0);
              pangolin::glDrawRectPerimeter(c[0] - radius, c[1] - radius,
                                            c[0] + radius, c[1] + radius);
            }

            if (show_ids)
              pangolin::GlFont::I().Text("%d", int(c[3])).Draw(c[0], c[1]);
          }
        }

        glColor3f(1.0, 0.0, 0.0);
        pangolin::GlFont::I()
            .Text("Tracked %d points", points.size())
            .Draw(5, 50);
      }
    }

    if (show_flow) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      if (vis_data) {
        const Eigen::aligned_map<granite::KeypointId, Eigen::AffineCompact2f>&
            kp_map = vis_data->opt_flow_res->observations.at(cam_id);
        const auto& pyr_levels =
            vis_data->opt_flow_res->pyramid_levels.at(cam_id);

        for (const auto& kv : kp_map) {
          const float scale = pyr_levels.at(kv.first) + 1.f;
          Eigen::MatrixXf transformed_patch =
              scale * kv.second.linear() * opt_flow_ptr->patch_coord;
          transformed_patch.colwise() += kv.second.translation();

          for (int i = 0; i < transformed_patch.cols(); i++) {
            const Eigen::Vector2f c = transformed_patch.col(i);
            pangolin::glDrawCirclePerimeter(c[0], c[1], 0.5f * scale);
          }

          const Eigen::Vector2f c = kv.second.translation();

          if (show_ids)
            pangolin::GlFont::I()
                .Text("%d", kv.first)
                .Draw((6.5f * scale) + c[0], (6.5f * scale) + c[1]);
        }

        pangolin::GlFont::I()
            .Text("%d opt_flow patches", kp_map.size())
            .Draw(5, 75);
      }
    }

    if (show_epipolar && calib.T_i_c.size() > 1) {
      glLineWidth(1.0);
      glColor3f(0.0, 1.0, 1.0);  // bright teal
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      Sophus::SE3d T_this_other =
          cam_id == 0 ? calib.T_i_c.at(0).inverse() * calib.T_i_c.at(1)
                      : calib.T_i_c.at(1).inverse() * calib.T_i_c.at(0);

      Eigen::Vector4d p0;
      p0.head<3>() = T_this_other.translation().normalized();
      p0(3) = 1.0;

      int line_id = 0;
      for (double i = -M_PI_2 / 2; i <= M_PI_2 / 2; i += 0.05) {
        Eigen::Vector4d p1(0, sin(i), cos(i), 1.0);

        if (cam_id == 0) p1.head<3>() = T_this_other * p1.head<3>();

        p1.normalize();

        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            line;
        for (double j = -1; j <= 1; j += 0.001) {
          Eigen::Vector2d proj;
          calib.intrinsics[cam_id].project(p0 * j + (1 - std::abs(j)) * p1,
                                           proj);
          line.emplace_back(proj);
        }

        Eigen::Vector2d c;
        calib.intrinsics[cam_id].project(p1, c);
        pangolin::GlFont::I().Text("%d", line_id).Draw(c[0], c[1]);
        line_id++;

        pangolin::glDrawLineStrip(line);
      }
    }
  }

  /// @brief Draws 3D scene.
  void draw_scene(pangolin::View& view) {
    UNUSED(view);
    view.Activate(camera);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    size_t frame_id = show_frame;
    int64_t t_ns = vio_dataset->get_image_timestamps().at(frame_id);

    if (show_gt_traj && !gt_t_ns.empty()) {
      const auto gt_now_T_w_i = interpolate_trajectory(gt_t_ns, gt_T_w_i, t_ns);

      glColor3ubv(gt_color);
      glLineWidth(4.f);

      if (calib.intrinsics.size() > 1 || use_imu) {
        pangolin::glDrawLineStrip(gt_t_w_i);
        pangolin::glDrawAxis(gt_now_T_w_i.matrix(), 0.2);
      } else {
        pangolin::glDrawLineStrip(gt_t_w_c0);
        pangolin::glDrawAxis((gt_now_T_w_i * calib.T_i_c.at(0)).matrix(), 0.2);
      }

      for (size_t i = 0; i < calib.T_i_c.size(); i++) {
        render_camera((gt_now_T_w_i * calib.T_i_c[i]).matrix(), 2.0f, gt_color,
                      0.1f);
      }
    }

    pangolin::ColourWheel colour_wheel;

    for (size_t map_idx = 0; map_idx < vio_t_ns.size(); map_idx++) {
      glPointSize(3);

      pangolin::Colour colour = colour_wheel.GetUniqueColour();

      glColor3f(colour.red, colour.green, colour.blue);
      glLineWidth(4.f);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      if ((calib.intrinsics.size() > 1 || use_imu) &&
          !vio_t_w_i.at(map_idx).empty()) {
        Eigen::aligned_vector<Eigen::Vector3d> sub_t_w_i;
        for (size_t idx = 0; idx < vio_t_ns.at(map_idx).size(); idx++) {
          if (vio_t_ns.at(map_idx).at(idx) <=
              vio_dataset->get_image_timestamps().at(show_frame)) {
            sub_t_w_i.push_back(T_gt_est.at(map_idx) *
                                vio_t_w_i.at(map_idx).at(idx));
          }
        }

        pangolin::glDrawLineStrip(sub_t_w_i);

      } else if (!vio_t_w_c0.at(map_idx).empty()) {
        Eigen::aligned_vector<Eigen::Vector3d> sub_t_w_c0;
        for (size_t idx = 0; idx < vio_t_ns.at(map_idx).size(); idx++) {
          if (vio_t_ns.at(map_idx).at(idx) <=
              vio_dataset->get_image_timestamps().at(show_frame)) {
            sub_t_w_c0.push_back(sT_gt_est.at(map_idx) *
                                 vio_t_w_c0.at(map_idx).at(idx));
          }
        }

        pangolin::glDrawLineStrip(sub_t_w_c0);
      }

      // may not be synchronous to vio_t_ns
      if (vis_map.size() > map_idx) {
        auto it = vis_map.at(map_idx).find(
            vio_dataset->get_image_timestamps()[frame_id]);

        if (it != vis_map.at(map_idx).end()) {
          if (!it->second->states.empty()) {
            const Sophus::SE3d T_w_i =
                T_gt_est.at(map_idx) * it->second->states.back();
            pangolin::glDrawAxis(T_w_i.matrix(), 0.3);
            for (size_t i = 0; i < calib.T_i_c.size(); i++) {
              render_camera((T_w_i * calib.T_i_c[i]).matrix(), 2.0f, cam_color,
                            0.1f);
            }
          } else if (!it->second->frames.empty()) {
            if (calib.intrinsics.size() > 1) {
              const Sophus::SE3d T_w_i =
                  T_gt_est.at(map_idx) * it->second->frames.back();
              pangolin::glDrawAxis(T_w_i.matrix(), 0.3);
              for (size_t i = 0; i < calib.T_i_c.size(); i++) {
                render_camera((T_w_i * calib.T_i_c[i]).matrix(), 2.0f,
                              cam_color, 0.1f);
              }
            } else {
              const Sophus::SE3d T_w_c0 =
                  sT_gt_est.at(map_idx) *
                  (it->second->frames.back() * calib.T_i_c[0]);
              pangolin::glDrawAxis(T_w_c0.matrix(), 0.3);
              render_camera(T_w_c0.matrix(), 2.0f, cam_color, 0.1f);
            }
          }

          for (const auto& p : it->second->states)
            for (size_t i = 0; i < calib.T_i_c.size(); i++)
              render_camera(
                  (T_gt_est.at(map_idx) * p * calib.T_i_c[i]).matrix(), 2.0f,
                  state_color, 0.1f);

          for (const auto& p : it->second->frames)
            if (calib.intrinsics.size() > 1) {
              for (size_t i = 0; i < calib.T_i_c.size(); i++)
                render_camera(
                    (T_gt_est.at(map_idx) * p * calib.T_i_c[i]).matrix(), 2.0f,
                    pose_color, 0.1f);
            } else {
              render_camera(
                  (sT_gt_est.at(map_idx) * (p * calib.T_i_c[0])).matrix(), 2.0f,
                  pose_color, 0.1f);
            }

          glColor3ubv(pose_color);
          Eigen::aligned_vector<Eigen::Vector3f> aligned_points(
              it->second->points.size());
          std::transform(it->second->points.cbegin(), it->second->points.cend(),
                         aligned_points.begin(),
                         [&](const Eigen::Vector3d& point) -> Eigen::Vector3f {
                           return (calib.intrinsics.size() > 1
                                       ? T_gt_est.at(map_idx) * point
                                       : sT_gt_est.at(map_idx) * point)
                               .cast<float>();
                         });
          pangolin::glDrawPoints(aligned_points);
        }
      }
    }

    pangolin::glDrawAxis(Sophus::SE3d().matrix(), 1.0);
  }

  void draw_plots() {
    plotter->ClearSeries();
    plotter->ClearMarkers();
    plotter->ResetColourWheel();

    if (show_est_pos) {
      plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est position x",
                         &vio_data_log);
      plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est position y",
                         &vio_data_log);
      plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est position z",
                         &vio_data_log);
    }

    if (show_est_vel) {
      plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est velocity x",
                         &vio_data_log);
      plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est velocity y",
                         &vio_data_log);
      plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est velocity z",
                         &vio_data_log);
    }

    if (show_est_bg) {
      plotter->AddSeries("$0", "$7", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est gyro bias x",
                         &vio_data_log);
      plotter->AddSeries("$0", "$8", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est gyro bias y",
                         &vio_data_log);
      plotter->AddSeries("$0", "$9", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est gyro bias z",
                         &vio_data_log);
    }

    if (show_est_ba) {
      plotter->AddSeries("$0", "$10", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est accel bias x",
                         &vio_data_log);
      plotter->AddSeries("$0", "$11", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est accel bias y",
                         &vio_data_log);
      plotter->AddSeries("$0", "$12", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "est accel bias z",
                         &vio_data_log);
    }

    if (show_gt_pos) {
      plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "gt position x",
                         &gt_data_log);
      plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "gt position y",
                         &gt_data_log);
      plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "gt position z",
                         &gt_data_log);
    }

    if (show_gt_vel) {
      plotter->AddSeries("$0", "$4", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "gt velocity x",
                         &gt_data_log);
      plotter->AddSeries("$0", "$5", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "gt velocity y",
                         &gt_data_log);
      plotter->AddSeries("$0", "$6", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "gt velocity z",
                         &gt_data_log);
    }

    if (show_scale) {
      plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "scale gt",
                         &vis_data_log);
      plotter->AddSeries("$0", "$1", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "scale variance",
                         &state_data_log);
      plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(), "drift variance",
                         &state_data_log);
    }

    if (show_entropy) {
      plotter->AddSeries("$0", "$2", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(),
                         "n entropy last frame", &vis_data_log);
      plotter->AddSeries("$0", "$3", pangolin::DrawingModeLine,
                         pangolin::Colour::Unspecified(),
                         "n avg entropy last frame", &vis_data_log);
    }

    if (show_kf) {
      for (size_t idx = 0; idx < vio_kf.size(); idx++) {
        if (vio_kf.at(idx)) {
          double t =
              (vio_dataset->get_image_timestamps()[idx] - start_t_ns) * 1e-9;
          plotter->AddMarker(pangolin::Marker::Vertical, t,
                             pangolin::Marker::Equal,
                             pangolin::Colour::Green());
        }
      }
    }

    double t =
        (vio_dataset->get_image_timestamps()[show_frame] - start_t_ns) * 1e-9;
    plotter->AddMarker(pangolin::Marker::Vertical, t, pangolin::Marker::Equal,
                       pangolin::Colour::White());
  }

  void loadCamCalib(const std::string& calib_path) {
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

  void populate_gt_data_log() {
    gt_data_log.Clear();
    for (size_t i = 0; i < vio_dataset->get_gt_timestamps().size(); i++) {
      std::vector<float> vals;
      vals.push_back((vio_dataset->get_gt_timestamps()[i] - start_t_ns) * 1e-9);
      for (int j = 0; j < 3; j++)
        vals.push_back(vio_dataset->get_gt_pose_data().at(i).translation()(j));

      if (!vio_dataset->get_gt_velocities().empty()) {
        for (int j = 0; j < 3; j++)
          vals.push_back(vio_dataset->get_gt_velocities().at(i)(j));
      }

      gt_data_log.Log(vals);
    }
  }

  void populate_vio_data_log() {
    vio_data_log.Clear();

    for (size_t map_idx = 0; map_idx < vio_t_ns.size(); map_idx++) {
      for (size_t idx = 0; idx < vio_t_ns.at(map_idx).size(); idx++) {
        std::vector<float> vals;
        vals.push_back((vio_t_ns.at(map_idx).at(idx) - start_t_ns) * 1e-9);

        Eigen::Vector3d v =
            calib.intrinsics.size() > 1
                ? T_gt_est.at(map_idx) * vio_vel_w_i.at(map_idx).at(idx)
                : sT_gt_est.at(map_idx) * vio_vel_w_i.at(map_idx).at(idx);

        Eigen::Vector3d t =
            (calib.intrinsics.size() > 1
                 ? T_gt_est.at(map_idx) * vio_T_w_i.at(map_idx).at(idx)
                 : sT_gt_est.at(map_idx) * vio_T_w_i.at(map_idx).at(idx))
                .translation();

        for (int i = 0; i < 3; i++) vals.push_back(v[i]);
        for (int i = 0; i < 3; i++) vals.push_back(t[i]);
        for (int i = 0; i < 3; i++)
          vals.push_back(vio_bg.at(map_idx).at(idx)[i]);
        for (int i = 0; i < 3; i++)
          vals.push_back(vio_ba.at(map_idx).at(idx)[i]);

        vio_data_log.Log(vals);
      }
    }
  }

  /// @brief Increments show_frame (if possible) and notifies image_feeder
  /// thread.
  ///
  /// @returns true if increment was possible otherwise false
  bool next_step() {
    if (show_frame < int(vio_dataset->get_image_timestamps().size()) - 1) {
      if (last_pushed_t_ns >=
          vio_dataset->get_image_timestamps().at(show_frame)) {
        show_frame = show_frame + 1;
        show_frame.Meta().gui_changed = true;
      }
      if (vio_t_ns.empty() ||
          vio_t_ns.back().back() <
              vio_dataset->get_image_timestamps().at(show_frame)) {
        cv.notify_one();
      }
      return true;
    } else {
      return false;
    }
  }

  /// @brief Decrements show_frame if possible
  ///
  /// @returns true if decrement was possible otherwise false
  bool prev_step() {
    if (show_frame > 0) {
      show_frame = show_frame - 1;
      show_frame.Meta().gui_changed = true;
      return true;
    } else {
      return false;
    }
  }

  /// @brief Computes SE3 (T_gt_est) and Sim3 (sT_gt_est) alignment for every
  /// map
  void alignButton() {
    for (size_t map_idx = 0; map_idx < vio_t_ns.size(); map_idx++) {
      std::vector<granite::FrameId> map_t_ns(vio_t_ns.at(map_idx).begin(),
                                            vio_t_ns.at(map_idx).end());

      if (calib.intrinsics.size() > 1 || use_imu) {
        Eigen::aligned_vector<Eigen::Vector3d> map_t_w_i(
            vio_t_w_i.at(map_idx).begin(), vio_t_w_i.at(map_idx).end());

        granite::alignSVD(map_t_ns, map_t_w_i, gt_t_ns, gt_t_w_i,
                         T_gt_est.at(map_idx), sT_gt_est.at(map_idx));
      } else {
        Eigen::aligned_vector<Eigen::Vector3d> map_t_w_c0(
            vio_t_w_c0.at(map_idx).begin(), vio_t_w_c0.at(map_idx).end());

        granite::alignSVD(map_t_ns, map_t_w_c0, gt_t_ns, gt_t_w_c0,
                         T_gt_est.at(map_idx), sT_gt_est.at(map_idx));
      }
    }
    populate_vio_data_log();
  }

  /// @brief saves the estimated trajectory to disk
  void saveTrajectoryButton() {
    std::cout << "Saving trajectory..." << std::endl;

    size_t best_idx = 0;
    {
      size_t best_length = 0;
      for (size_t idx = 0; idx < vio_t_ns.size(); idx++) {
        if (vio_t_ns.at(idx).size() > best_length) {
          best_idx = idx;
          best_length = vio_t_ns.at(idx).size();
        }
      }
    }

    tbb::concurrent_vector<tbb::concurrent_vector<
        Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>>& vio_T =
        vio_T_w_i;

    if (calib.T_i_c.size() < 2 && !use_imu) {
      vio_T = vio_T_w_c0;
    }

    if (tum_rgbd_fmt) {
      std::ofstream os("trajectory.txt");

      if (calib.T_i_c.size() < 2 && !use_imu) {
        os << "# trajectory is in camera coordinate frame" << std::endl;
      } else {
        os << "# trajectory is in IMU coordinate frame" << std::endl;
      }
      os << "# timestamp tx ty tz qx qy qz qw" << std::endl;

      for (size_t i = 0; i < vio_t_ns.at(best_idx).size(); i++) {
        const Sophus::SE3d& pose = vio_T.at(best_idx)[i];
        os << std::scientific << std::setprecision(18)
           << vio_t_ns.at(best_idx)[i] * 1e-9 << " " << pose.translation().x()
           << " " << pose.translation().y() << " " << pose.translation().z()
           << " " << pose.unit_quaternion().x() << " "
           << pose.unit_quaternion().y() << " " << pose.unit_quaternion().z()
           << " " << pose.unit_quaternion().w() << std::endl;
      }

      os.close();

      std::cout
          << "Saved trajectory in TUM RGB-D Dataset format in trajectory.txt"
          << std::endl;
    } else if (euroc_fmt) {
      std::ofstream os("trajectory.csv");

      if (calib.T_i_c.size() < 2 && !use_imu) {
        os << "# trajectory is in camera coordinate frame" << std::endl;
      } else {
        os << "# trajectory is in IMU coordinate frame" << std::endl;
      }
      os << "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w "
            "[],q_RS_x [],q_RS_y [],q_RS_z []"
         << std::endl;

      for (size_t i = 0; i < vio_t_ns.at(best_idx).size(); i++) {
        const Sophus::SE3d& pose = vio_T.at(best_idx)[i];
        os << std::scientific << std::setprecision(18)
           << vio_t_ns.at(best_idx)[i] << "," << pose.translation().x() << ","
           << pose.translation().y() << "," << pose.translation().z() << ","
           << pose.unit_quaternion().w() << "," << pose.unit_quaternion().x()
           << "," << pose.unit_quaternion().y() << ","
           << pose.unit_quaternion().z() << std::endl;
      }

      std::cout << "Saved trajectory in Euroc Dataset format in trajectory.csv"
                << std::endl;
    } else {
      std::ofstream os("trajectory_kitti.txt");

      for (size_t i = 0; i < vio_t_ns.at(best_idx).size(); i++) {
        Eigen::Matrix<double, 3, 4> mat = vio_T.at(best_idx)[i].matrix3x4();
        os << std::scientific << std::setprecision(12) << mat.row(0) << " "
           << mat.row(1) << " " << mat.row(2) << " " << std::endl;
      }

      os.close();

      std::cout
          << "Saved trajectory in KITTI Dataset format in trajectory_kitti.txt"
          << std::endl;
    }
  }

  /// @brief Data structure for write_stats
  struct StatsData {
    static constexpr float NaN = std::numeric_limits<float>::quiet_NaN();

    float est_pos_x = NaN;
    float est_pos_y = NaN;
    float est_pos_z = NaN;

    float est_rot_x = NaN;
    float est_rot_y = NaN;
    float est_rot_z = NaN;
    float est_rot_w = NaN;

    float est_vel_x = NaN;
    float est_vel_y = NaN;
    float est_vel_z = NaN;

    float est_bg_x = NaN;
    float est_bg_y = NaN;
    float est_bg_z = NaN;

    float est_ba_x = NaN;
    float est_ba_y = NaN;
    float est_ba_z = NaN;

    float gt_pos_x = NaN;
    float gt_pos_y = NaN;
    float gt_pos_z = NaN;

    float gt_rot_x = NaN;
    float gt_rot_y = NaN;
    float gt_rot_z = NaN;
    float gt_rot_w = NaN;

    float gt_vel_x = NaN;
    float gt_vel_y = NaN;
    float gt_vel_z = NaN;

    float gt_scale = NaN;
    float scale_variance = NaN;
    float drift_variance = NaN;

    float n_entropy_last_frame = NaN;
    float avg_n_entropy_last_frame = NaN;
  };

  /// @brief Writes all collected data into one file. The resulting file can be
  /// parsed to e.g. draw plots
  void write_stats(const std::string& file_path) {
    std::map<float, StatsData> stats_map;
    {
      size_t total_idx = 0;
      for (size_t map_idx = 0; map_idx < vio_t_ns.size(); map_idx++) {
        for (size_t idx = 0; idx < vio_t_ns.at(map_idx).size(); idx++) {
          float t_s = vio_t_ns.at(map_idx).at(idx) * 1e-9;

          const Sophus::SE3d pose =
              (calib.intrinsics.size() > 1
                   ? T_gt_est.at(map_idx) * vio_T_w_i.at(map_idx).at(idx)
                   : sT_gt_est.at(map_idx) * vio_T_w_i.at(map_idx).at(idx));

          const Eigen::Vector3d vel =
              calib.intrinsics.size() > 1
                  ? T_gt_est.at(map_idx) * vio_vel_w_i.at(map_idx).at(idx)
                  : sT_gt_est.at(map_idx) * vio_vel_w_i.at(map_idx).at(idx);

          // const Sophus::SE3d pose = vio_T_w_i.at(idx);
          //
          // const Eigen::Vector3d vel = vio_vel_w_i.at(idx);

          stats_map[t_s].est_pos_x = pose.translation().x();
          stats_map[t_s].est_pos_y = pose.translation().y();
          stats_map[t_s].est_pos_z = pose.translation().z();

          stats_map[t_s].est_rot_x = pose.unit_quaternion().x();
          stats_map[t_s].est_rot_y = pose.unit_quaternion().y();
          stats_map[t_s].est_rot_z = pose.unit_quaternion().z();
          stats_map[t_s].est_rot_w = pose.unit_quaternion().w();

          stats_map[t_s].est_vel_x = vel.x();
          stats_map[t_s].est_vel_y = vel.y();
          stats_map[t_s].est_vel_z = vel.z();

          stats_map[t_s].est_bg_x = vio_bg.at(map_idx).at(idx).x();
          stats_map[t_s].est_bg_y = vio_bg.at(map_idx).at(idx).y();
          stats_map[t_s].est_bg_z = vio_bg.at(map_idx).at(idx).z();

          stats_map[t_s].est_ba_x = vio_ba.at(map_idx).at(idx).x();
          stats_map[t_s].est_ba_y = vio_ba.at(map_idx).at(idx).y();
          stats_map[t_s].est_ba_z = vio_ba.at(map_idx).at(idx).z();

          stats_map[t_s].gt_scale = scale_vec.at(total_idx);
          stats_map[t_s].scale_variance = scale_variance.at(total_idx);
          stats_map[t_s].drift_variance = drift_variance.at(total_idx);
          stats_map[t_s].n_entropy_last_frame =
              n_entropy_last_frame_vec.at(total_idx);
          stats_map[t_s].avg_n_entropy_last_frame =
              avg_n_entropy_last_frame_vec.at(total_idx);
          total_idx++;
        }
      }
    }
    {
      for (size_t i = 0; i < vio_dataset->get_gt_timestamps().size(); i++) {
        float t_s = vio_dataset->get_gt_timestamps()[i] * 1e-9;

        // const Sophus::SE3d pose =
        //     (calib.intrinsics.size() > 1
        //          ? T_gt_est.inverse() * vio_dataset->get_gt_pose_data().at(i)
        //          : sT_gt_est.inverse() *
        //          vio_dataset->get_gt_pose_data().at(i));

        const Sophus::SE3d pose = vio_dataset->get_gt_pose_data().at(i);

        stats_map[t_s].gt_pos_x = pose.translation().x();
        stats_map[t_s].gt_pos_y = pose.translation().y();
        stats_map[t_s].gt_pos_z = pose.translation().z();

        stats_map[t_s].gt_rot_x = pose.unit_quaternion().x();
        stats_map[t_s].gt_rot_y = pose.unit_quaternion().y();
        stats_map[t_s].gt_rot_z = pose.unit_quaternion().z();
        stats_map[t_s].gt_rot_w = pose.unit_quaternion().w();

        if (!vio_dataset->get_gt_velocities().empty()) {
          // const Eigen::Vector3d vel =
          //     calib.intrinsics.size() > 1
          //         ? T_gt_est.inverse() *
          //         vio_dataset->get_gt_velocities().at(i) :
          //         sT_gt_est.inverse() *
          //         vio_dataset->get_gt_velocities().at(i);

          const Eigen::Vector3d& vel = vio_dataset->get_gt_velocities().at(i);

          stats_map[t_s].gt_vel_x = vel.x();
          stats_map[t_s].gt_vel_y = vel.y();
          stats_map[t_s].gt_vel_z = vel.z();
        }
      }
    }

    std::ofstream os(file_path);
    os << std::setprecision(15);
    os << ",t_s,est_pos_x,est_pos_y,est_pos_z,est_rot_x,est_rot_y,est_rot_z,"
          "est_rot_w,est_vel_x,est_vel_y,est_vel_z,est_bg_x,est_bg_y,est_bg_z,"
          "est_ba_x,est_ba_y,est_ba_z,gt_pos_x,gt_pos_y,gt_pos_z,gt_rot_x,gt_"
          "rot_y,gt_rot_z,gt_rot_w,gt_vel_x,gt_vel_y,gt_vel_z,gt_scale,scale_"
          "variance,drift_variance,n_entropy_last_frame,avg_n_entropy_last_"
          "frame"
       << '\n';

    int line = 1;
    for (const auto& stats_kv : stats_map) {
      os << line << ',' << stats_kv.first << ',' << stats_kv.second.est_pos_x
         << ',' << stats_kv.second.est_pos_y << ',' << stats_kv.second.est_pos_z
         << ',' << stats_kv.second.est_rot_x << ','

         << stats_kv.second.est_rot_y << ',' << stats_kv.second.est_rot_z << ','
         << stats_kv.second.est_rot_w << ','

         << stats_kv.second.est_vel_x << ',' << stats_kv.second.est_vel_y << ','
         << stats_kv.second.est_vel_z << ','

         << stats_kv.second.est_bg_x << ',' << stats_kv.second.est_bg_y << ','
         << stats_kv.second.est_bg_z << ',' << stats_kv.second.est_ba_x << ','
         << stats_kv.second.est_ba_y << ',' << stats_kv.second.est_ba_z << ','

         << stats_kv.second.gt_pos_x << ',' << stats_kv.second.gt_pos_y << ','
         << stats_kv.second.gt_pos_z << ','

         << stats_kv.second.gt_rot_x << ',' << stats_kv.second.gt_rot_y << ','
         << stats_kv.second.gt_rot_z << ',' << stats_kv.second.gt_rot_w << ','

         << stats_kv.second.gt_vel_x << ',' << stats_kv.second.gt_vel_y << ','
         << stats_kv.second.gt_vel_z << ','

         << stats_kv.second.gt_scale << ',' << stats_kv.second.scale_variance
         << ',' << stats_kv.second.drift_variance << ','
         << stats_kv.second.n_entropy_last_frame << ','
         << stats_kv.second.avg_n_entropy_last_frame

         << '\n';
      line++;
    }

    os.close();
  }

  // GUI constants
  static constexpr int PANEL_WIDTH = 200;

  // GUI variables
  bool show_gui = true;
  std::atomic<bool> terminate = false;
  size_t last_frame_processed = 0;
  std::mutex m;
  std::condition_variable cv;
  bool step_by_step = false;
  std::atomic<granite::FrameId> last_pushed_t_ns = -1;

  // GUI settings
  pangolin::Var<int> show_frame;

  pangolin::Var<bool> show_flow;
  pangolin::Var<bool> show_obs;
  pangolin::Var<bool> show_ids;
  pangolin::Var<bool> show_epipolar;

  pangolin::Var<bool> show_kf;
  pangolin::Var<bool> show_est_pos;
  pangolin::Var<bool> show_est_vel;
  pangolin::Var<bool> show_est_bg;
  pangolin::Var<bool> show_est_ba;

  pangolin::Var<bool> show_gt_traj;
  pangolin::Var<bool> show_gt_pos;
  pangolin::Var<bool> show_gt_vel;
  pangolin::Var<bool> show_scale;
  pangolin::Var<bool> show_entropy;

  pangolin::Var<bool> continue_btn;
  pangolin::Var<bool> continue_fast;

  pangolin::Var<bool> euroc_fmt;
  pangolin::Var<bool> tum_rgbd_fmt;
  pangolin::Var<bool> kitti_fmt;

  pangolin::Var<bool> follow;
  // pangolin::Var<bool> record("ui.record", false, false, true);

  // GUI objects
  pangolin::OpenGlRenderState camera;
  pangolin::Plotter* plotter;
  pangolin::DataLog imu_data_log, vio_data_log, error_data_log, gt_data_log,
      vis_data_log, state_data_log;
  pangolin::View* display3D;
  pangolin::View* img_view_display;
  std::vector<std::shared_ptr<pangolin::ImageView>> img_view;

  // statistics
  bool print_queue = false;
  std::string trajectory_fmt;
  std::string result_path;
  std::string stats_path;
  std::vector<bool> vio_kf;
  tbb::concurrent_vector<
      tbb::concurrent_unordered_map<int64_t, granite::VioVisualizationData::Ptr>>
      vis_map;
  int64_t start_t_ns;
  tbb::concurrent_vector<tbb::concurrent_vector<int64_t>> vio_t_ns;
  tbb::concurrent_vector<tbb::concurrent_vector<
      Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>>
      vio_t_w_i, vio_t_w_c0;

  tbb::concurrent_vector<tbb::concurrent_vector<
      Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>>
      vio_T_w_i, vio_T_w_c0;

  tbb::concurrent_vector<tbb::concurrent_vector<
      Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>>
      vio_vel_w_i, vio_bg, vio_ba;

  std::vector<double> scale_vec, scale_variance, drift_variance,
      n_entropy_last_frame_vec, avg_n_entropy_last_frame_vec;
  std::vector<int64_t> gt_t_ns;
  tbb::concurrent_unordered_map<int64_t, int, std::hash<int64_t>>
      timestamp_to_id;

  Eigen::aligned_vector<Sophus::SE3d> gt_T_w_i;
  Eigen::aligned_vector<Sophus::SE3d> gt_T_w_c0;
  Eigen::aligned_vector<Eigen::Vector3d> gt_t_w_i;
  Eigen::aligned_vector<Eigen::Vector3d> gt_t_w_c0;

  tbb::concurrent_vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>
      T_gt_est;
  tbb::concurrent_vector<Sophus::Sim3d, Eigen::aligned_allocator<Sophus::Sim3d>>
      sT_gt_est;

  // VIO system
  std::string marg_data_path;
  granite::MargDataSaver::Ptr marg_data_saver;
  tbb::concurrent_bounded_queue<granite::ImuData::Ptr> imu_data_queue;
  tbb::concurrent_bounded_queue<granite::VioVisualizationData::Ptr>
      out_vis_queue;
  tbb::concurrent_bounded_queue<granite::VioStateData::Ptr> out_state_queue;

  std::shared_ptr<std::thread> image_feeder, imu_feeder, vis_data_receiver,
      state_receiver, queue_size_printer;

  granite::Calibration<double> calib;

  granite::VioDatasetPtr vio_dataset;
  granite::VioConfig vio_config;
  bool use_imu = true;
  granite::OpticalFlowBase::Ptr opt_flow_ptr;
  granite::VioEstimatorBase::Ptr vio;
};

int main(int argc, char** argv) {
  bool show_gui = true;
  bool print_queue = false;
  bool step_by_step = false;
  bool use_imu = true;
  std::string cam_calib_path;
  std::string dataset_path;
  std::string dataset_type;
  std::string config_path;
  std::string result_path;
  std::string stats_path;
  std::string trajectory_fmt;
  std::string marg_data_path;
  int num_threads = 0;

  CLI::App app{"App description"};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--cam-calib", cam_calib_path,
                 "Ground-truth camera calibration used for simulation.")
      ->required();

  app.add_option("--dataset-path", dataset_path, "Path to dataset.")
      ->required();

  app.add_option("--dataset-type", dataset_type, "Dataset type <euroc, bag>.")
      ->required();

  app.add_option("--marg-data", marg_data_path,
                 "Path to folder where marginalization data will be stored.");

  app.add_option("--print-queue", print_queue, "Print queue.");
  app.add_option("--config-path", config_path, "Path to config file.");
  app.add_option("--result-path", result_path,
                 "Path to result file where the system will write RMSE ATE.");
  app.add_option("--stats-path", stats_path,
                 "Path to stats file where the system will write some stats.");
  app.add_option("--num-threads", num_threads, "Number of threads.");
  app.add_option("--step-by-step", step_by_step, "Do not start automatically.");
  app.add_option("--save-trajectory", trajectory_fmt,
                 "Save trajectory. Supported formats <tum, euroc, kitti>");
  app.add_option("--use-imu", use_imu, "Use IMU.");

  // global thread limit is in effect until global_control object is destroyed
  std::unique_ptr<tbb::global_control> tbb_global_control;
  if (num_threads > 0) {
    tbb_global_control = std::make_unique<tbb::global_control>(
        tbb::global_control::max_allowed_parallelism, num_threads);
  }

  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  if (!show_gui) {
    std::cout
        << "You tried to start the system without GUI but without autostart."
        << std::endl;
    step_by_step = false;
  }

  VioDataset vd(cam_calib_path, config_path, dataset_type, dataset_path,
                marg_data_path, result_path, stats_path, trajectory_fmt,
                show_gui, use_imu, step_by_step, print_queue);

  vd.run();

  std::cout << "Finish" << std::endl;

  return 0;
}
