## MADMAX dataset

Go to [Arches MADMAX dataset](https://datasets.arches-projekt.de/morocco2018/) and download the sequences you want to evaluate. Extract the archives in such a way that you get the following folder structure:

```
  |- A_0
      |- ground_truth
      |- rect_left
      |- rect_right
      |- A-0_imu_data.csv
  |- A_1
...
```

We provide a script to establish an EuRoC like folder structure.

```
scripts/convert_madmax_to_euroc.py --dataset_path path/to/MADMAX/A-0
```

To run the visual odometry execute:

```
granite_vio --dataset-path path/to/MADMAX/A-0 --cam-calib data/madmax_calib_mono.json --dataset-type euroc --config-path data/madmax_config_mono.json --use-imu 0 --show-gui 1 --step-by-step 1
```

![madmax_vio](/doc/img/madmax_vio.png)


## KITTI dataset

[![teaser](/doc/img/kitti_video.png)](https://www.youtube.com/watch?v=M_ZcNgExUNc)

We demonstrate the usage of the system with the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) as an example.

Download the sequences (`data_odometry_gray.zip`) from the dataset and extract it. 
```
# We assume you have extracted the sequences in ~/dataset_gray/sequences/
# Convert calibration to the granite format
granite_convert_kitti_calib.py -d ~/dataset_gray/sequences/00/

# If you want to convert calibrations for all sequences use the following command
for i in {00..21}; do granite_convert_kitti_calib.py -d ~/dataset_gray/sequences/$i/; done
```
Optionally you can also copy the provided ground-truth poses to `poses.txt` in the corresponding sequence.

To run the visual odometry execute the following command.
```
granite_vio --dataset-path ~/dataset_gray/sequences/00/ --cam-calib /work/kitti/dataset_gray/sequences/00/granite_calib.json --dataset-type kitti --config-path data/kitti_config.json --show-gui 1 --use-imu 0
```
![magistrale1_vio](/doc/img/kitti.png)

## Other Datasets

Of course the VO can also be run on the VIO Datasets described in [VioMapping](VioMapping.md)
