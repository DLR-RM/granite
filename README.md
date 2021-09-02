## Granite

This project contains the code for the IEEE IROS 2021 paper **Towards Robust Monocular Visual Odometry for Flying Robots on Planetary Missions**, M. Wudenka, M. G. Müller, N. Demmel, A. Wedler, R. Triebel, D. Cremers, W. Stürzl.

It is mainly build on [basalt](https://vision.in.tum.de/research/vslam/basalt).

Main differences are:

* support of monocular setups for visual odometry (not mapping)
* new keyframe selection strategy based on negative entropy for non-inertial setups
* integration of landmarks "at infinity" to the optimization
* detection of features to track on more image pyramid levels
* computation of a scalar indicator for scale variance (stereo) or drift variance (monocular)

## Installation

### Source installation for Ubuntu >= 18.04 and MacOS >= 10.14 Mojave
Clone the source code for the project and build it. For MacOS you should have [Homebrew](https://brew.sh/) installed.
```
git clone --recursive https://github.com/DLR-RM/granite.git
cd granite
./scripts/install_deps.sh
```

Go to `include/granite/utils/common_types.h` and decide if the `PixelType` should be `uint8_t` or `uint16_t`. The `PixelType` determines how many bytes granite uses internally for image representation.

```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8
```
NOTE: It is possible to compile the code on Ubuntu 16.04, but you need to install cmake-3.10 or higher and gcc-7. See corresponding [Dockerfile](docker/b_image_xenial/Dockerfile) as an example.

## Usage
* [Camera, IMU and Mocap calibration. (TUM-VI, Euroc, UZH-FPV and Kalibr datasets)](doc/Calibration.md)
* [Visual odometry (no IMU). (MADMAX and KITTI dataset)](doc/Vo.md)
* [Visual-inertial odometry and mapping. (TUM-VI and Euroc datasets)](doc/VioMapping.md)
* [Simulation tools to test different components of the system.](doc/Simulation.md)

## Device support
* [Tutorial on Camera-IMU and Motion capture calibration with Realsense T265.](doc/Realsense.md)

## Development
* [Development environment setup.](doc/DevSetup.md)

## Licence

The source code is provided under a MIT license. See the LICENSE file for details.
Note also the different licenses of thirdparty submodules.

The original source code from Basalt was distributed under a BSD 3-clause license.