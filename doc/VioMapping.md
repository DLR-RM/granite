## EuRoC dataset

We demonstrate the usage of the system with the `MH_05_difficult` sequence of the [EuRoC dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) as an example.

Download the sequence from the dataset and extract it. 
```
mkdir euroc_data
cd euroc_data
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip
mkdir MH_05_difficult
cd MH_05_difficult
unzip ../MH_05_difficult.zip
cd ../
```

### Visual-inertial odometry
To run the visual-inertial odometry execute the following command in `euroc_data` folder where you downloaded the dataset.
```
granite_vio --dataset-path MH_05_difficult/ --cam-calib data/euroc_ds_calib_stereo.json --dataset-type euroc --config-path data/euroc_config_stereo.json --marg-data euroc_marg_data --show-gui 1 --use-imu 1
```
The command line options have the following meaning:
* `--dataset-path` path to the dataset.
* `--dataset-type` type of the dataset. Currently only `bag` and `euroc` formats of the datasets are supported.
* `--cam-calib` path to camera calibration file. Check [calibration instructions](doc/Calibration.md) to see how the calibration was generated.
* `--config-path` path to the configuration file.
* `--marg-data` folder where the data from keyframe marginalization will be stored. This data can be later used for visual-inertial mapping. Saving the marginalization data may slow down the system.
* `--show-gui` enables or disables GUI.
* `--stats-path` path to a file were the system will write extensive data of the run.
* `--use-imu` fuse inertial data.

This opens the GUI and runs the sequence. The processing happens in the background as fast as possible, and the visualization results are saved in the GUI and can be analysed offline.
![MH_05_VIO](/doc/img/MH_05_VIO.png)

The buttons in the GUI have the following meaning:
* `show_obs` toggles the visibility of the tracked landmarks in the image view.
* `show_ids` toggles the IDs of the points.
* `show_epipolar` toggles the epipolar lines.
* `show_est_pos` shows the plot of the estimated position.
* `show_est_vel` shows the plot of the estimated velocity.
* `show_est_bg` shows the plot of the estimated gyro bias.
* `show_est_ba` shows the plot of the estimated accel bias.
* `show_gt_pos` shows the plot og the ground truth position (if available).
* `show_gt_vel` shows the plot of the ground truth velocity (if available).
* `show_scale` shows the plots of the scaling factor that is needed to align the current optimization window to the ground truth trajectory (scale gt), the estimated variance in the scale for stereo systems (scale variance) and the estimated drift variance for monocular systems (drift variance)
* `show_entropy` shows the current and the sliding window average negative entropy of the last frame processed. Those values are usually between 80 and 100.
* `show_gt_traj` shows ground-truth trajectory in the 3D view.


By default the system starts with `continue_fast` enabled. This option visualizes the latest processed frame until the end of the sequence. Alternatively, the `continue` visualizes every frame without skipping. If both options are disabled the system shows the frame that is selected with the `show_frame` slider and the user can move forward and backward with `next_step` and `prev_step` buttons. The `follow` button changes between the static camera and the camera attached to the current frame.

For evaluation the button `align` is used. It aligns the GT trajectory with the current estimate using an SE(3) or Sim(3) transformation and prints the transformation and the root-mean-squared absolute trajectory error (RMS ATE).

The button `save_traj` saves the trajectory in one of two formats (`euroc_fmt` or `tum_rgbd_fmt`). In EuRoC format each pose is a line in the file and has the following format `timestamp[ns],tx,ty,tz,qw,qx,qy,qz`. TUM RBG-D can be used with [TUM RGB-D](https://vision.in.tum.de/data/datasets/rgbd-dataset/tools) or [UZH](https://github.com/uzh-rpg/rpg_trajectory_evaluation) trajectory evaluation tools and has the following format `timestamp[s] tx ty tz qx qy qz qw`. 

Currently only stereo, stereo-inertial and monocular odometry setups are supported.
Monocular-inertial setups can not be handled.

### Visual-inertial mapping
To run the mapping tool execute the following command:
```
granite_mapper --cam-calib data/euroc_ds_calib_stereo.json --marg-data euroc_marg_data
```
Here `--marg-data` is the folder with the results from VIO.

This opens the GUI and extracts non-linear factors from the marginalization data.
![MH_05_MAPPING](/doc/img/MH_05_MAPPING.png)

The buttons in the GUI have the following meaning:
* `show_frame1`, `show_cam1`, `show_frame2`, `show_cam2` allows you to assign images to image view 1 and 2 from different timestamps and cameras.
* `show_detected` shows the detected keypoints in the image view window.
* `show_matches` shows feature matching results.
* `show_inliers` shows inlier matches after geometric verification.
* `show_ids` prints point IDs. Can be used to find the same point in two views to check matches and inliers.
* `show_gt_traj` shows the ground-truth trajectory.
* `show_edges` shows the edges from the factors. Relative-pose factors in red, roll-pitch factors in magenta and bundle adjustment co-visibility edges in green.
* `show_points` shows 3D landmarks.

The workflow for the mapping is the following:
* `detect` detect the keypoints in the keyframe images.
* `match` run the geometric 2D to 2D matching between image frames.
* `tracks` build tracks from 2D matches and triangulate the points.
* `optimize` run the optimization.
* `align_se3` align ground-truth trajectory in SE(3) and print the transformation and the error.

The `num_opt_iter` slider controls the maximum number of iterations executed when pressing `optimize`.

The button `save_traj` works similar to the VIO, but saves the keyframe trajectory (subset of frames).

For more systematic evaluation see the evaluation scripts in the [scripts/eval_full](/scripts/eval_full) folder.

**NOTE: It appears that only the datasets in ASL Dataset Format (`euroc` dataset type in our notation) contain ground truth that is time-aligned to the IMU and camera images. It is located in the `state_groundtruth_estimate0` folder. Bag files have raw Mocap measurements that are not time aligned and should not be used for evaluations.**



### Optical Flow
The visual-inertial odometry relies on the optical flow results. To enable a better analysis of the system we also provide a separate optical flow executable
```
granite_opt_flow --dataset-path MH_05_difficult/ --cam-calib data/euroc_ds_calib_stereo.json --dataset-type euroc --config-path data/euroc_config_stereo.json --show-gui 1
```

This will run the GUI and print an average track length after the dataset is processed.
![MH_05_OPT_FLOW](/doc/img/MH_05_OPT_FLOW.png)


## TUM-VI dataset

We demonstrate the usage of the system with the `magistrale1` sequence of the [TUM-VI dataset](https://vision.in.tum.de/data/datasets/visual-inertial-dataset) as an example.

Download the sequence from the dataset and extract it. 
```
mkdir tumvi_data
cd tumvi_data
wget http://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-magistrale1_512_16.tar
tar -xvf dataset-magistrale1_512_16.tar
```

### Visual-inertial odometry
To run the visual-inertial odometry execute the following command in `tumvi_data` folder where you downloaded the dataset.
```
granite_vio --dataset-path dataset-magistrale1_512_16/ --cam-calib data/tumvi_512_ds_calib_stereo.json --dataset-type euroc --config-path data/tumvi_512_config_stereo.json --marg-data tumvi_marg_data --show-gui 1 
```
![magistrale1_vio](/doc/img/magistrale1_vio.png)

### Visual-inertial mapping
To run the mapping tool execute the following command:
```
granite_mapper --cam-calib data/tumvi_512_ds_calib_stereo.json --marg-data tumvi_marg_data
```
![magistrale1_mapping](/doc/img/magistrale1_mapping.png)

## Parameter Description

* **config.optical_flow_type** ["frame_to_frame", "patch"] Type of optical flow 
* **config.optical_flow_detection_grid_size** [integer] The optical flow will try to detect one corner per grid cell. Lower numbers lead to more features.
* **config.optical_flow_max_recovered_dist2** [float] Threshold for back and forth optical flow tracking. Features that do not return to a circle with that radius around the starting point are considered failures.
* **config.optical_flow_pattern** [24, 50, 51, 52] The patch coordinate pattern. See `include/granite/optical_flow/patterns.h`.
* **config.optical_flow_max_iterations** [integer] Maximum number of iterations the optimization of the SE(2) optical flow is executed.
* **config.optical_flow_epipolar_error** [float] Threshold distance of a optical flow result between stereo camera pairs to the epipolar curve. 
* **config.optical_flow_levels** [integer] Number of image pyramid levels used.
* **config.optical_flow_skip_frames** [integer] 1: The optical flow result of every frame is published; 2: Only every second optical flow result is published; ... .

* **config.vio_max_states** [integer] Only used in visual-inertial tracking. Number of recent frames kept in the optimization window with full state (velocity, bias, ...).
* **config.vio_max_kfs** [integer] Maximum number of keyframes kept in the optimization window.
* **config.vio_min_frames_after_kf** [integer] [deprecated] Currently only used in visual-inertial tracking. Minimum number of frames that have to pass by until a new keyframe is created.
* **config.vio_fraction_entropy_take_kf** [float] Threshold fraction of the running average negative entropy (see Kuo et.al. 2020) at which a new keyframe is created. Currently only used in visual tracking.
* **config.vio_new_kf_keypoints_thresh** [float] [deprecated] Currently only used in visual-inertial tracking. Threshold fraction of how many features in a frame should be associated to a landmark. If below, a new keyframe is created.
* **config.vio_debug** [bool] Prints additional information to the console.
* **config.vio_obs_std_dev** [float] Diagonal entries of the feature covariance matrix.
* **config.vio_obs_huber_thresh** [float] Huber parameter used in bundle adjustment.
* **config.vio_no_motion_reg_weight** [float] Currently only used in visual tracking. Weight of prior that ties the relative translation of frames that observe a landmark with inverse distance = 0 to zero.
* **config.vio_min_triangulation_dist** [float] Minimum threshold distance that has to be between to observing frames, when attempting triangulation. TODO: Replace with threshold on the angle between the bearing vectors.
* **config.vio_max_inverse_distance** [float] Threshold on the inverse distance. If a point has an inverse distance above after triangulation it is discarded.
* **config.vio_outlier_threshold** [float] Maximum error a reprojection residual can have. Observations with a higher error are removed from the optimization problem during outlier filtering.
* **config.vio_take_keyframe_iteration** [integer] Currently only used in visual tracking. The decision to take a new keyframe is made after this optimization iteration. Thus, the H matrix of this iteration is investigated.
* **config.vio_filter_iteration** [integer] After this iteration the outlier filtering based on the reprojection error is executed.
* **config.vio_max_iterations** [integer] Maximum number of GN or LM iterations executed. 
* **config.vio_enforce_realtime** [bool] Only use the youngest optical flow result. Discard older ones.
* **config.vio_use_lm** [bool] false: use Gauss-Newton; true: use Levenberg-Marquardt
* **config.vio_lm_lambda_min** [float] Trust region of Levenberg Marquardt
* **config.vio_lm_lambda_max** [float] Trust region of Levenberg Marquardt
* **config.vio_init_pose_weight** [float] Number used to initialize the marginalization H matrix in order to fix the absolute pose.
* **config.vio_mono_init_max_ransac_iter** [integer] Number of RANSAC iterations executed during monocular initialization.
* **config.vio_mono_init_min_parallax** [float] 20 points with a minimum parallax of this are needed to attempt a monocular map initialization.
* **config.vio_init_ba_weight** [float] Weight factor to keep the accelerometer bias nearly constant.
* **config.vio_init_bg_weight** [float] Weight factor to keep the gyroscope bias nearly constant.