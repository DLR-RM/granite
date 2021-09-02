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
https://gitlab.com/VladyslavUsenko/granite-headers.git

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

@file
@brief Serialization for granite types
*/

#pragma once

#include <granite/image/image.h>
#include <granite/serialization/eigen_io.h>
#include <granite/calibration/calibration.hpp>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/deque.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace cereal {

template <class Archive, class Scalar>
inline void save(Archive& ar, const granite::GenericCamera<Scalar>& cam) {
  std::visit(
      [&](const auto& cam) {
        ar(cereal::make_nvp("camera_type", cam.getName()));
        ar(cereal::make_nvp("intrinsics", cam));
      },
      cam.variant);
}

template <class Archive, class Scalar>
inline void load(Archive& ar, granite::GenericCamera<Scalar>& cam) {
  std::string cam_type;
  ar(cereal::make_nvp("camera_type", cam_type));

  cam = granite::GenericCamera<Scalar>::fromString(cam_type);

  std::visit([&](auto& cam) { ar(cereal::make_nvp("intrinsics", cam)); },
             cam.variant);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const granite::KannalaBrandtCamera4<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]),
     cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]),
     cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("k1", cam.getParam()[4]),
     cereal::make_nvp("k2", cam.getParam()[5]),
     cereal::make_nvp("k3", cam.getParam()[6]),
     cereal::make_nvp("k4", cam.getParam()[7]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, granite::KannalaBrandtCamera4<Scalar>& cam) {
  Eigen::Matrix<Scalar, 8, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]),
     cereal::make_nvp("cx", intr[2]), cereal::make_nvp("cy", intr[3]),
     cereal::make_nvp("k1", intr[4]), cereal::make_nvp("k2", intr[5]),
     cereal::make_nvp("k3", intr[6]), cereal::make_nvp("k4", intr[7]));

  cam = granite::KannalaBrandtCamera4<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar,
                 const granite::ExtendedUnifiedCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]),
     cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]),
     cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("alpha", cam.getParam()[4]),
     cereal::make_nvp("beta", cam.getParam()[5]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, granite::ExtendedUnifiedCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 6, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]),
     cereal::make_nvp("cx", intr[2]), cereal::make_nvp("cy", intr[3]),
     cereal::make_nvp("alpha", intr[4]), cereal::make_nvp("beta", intr[5]));

  cam = granite::ExtendedUnifiedCamera<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const granite::UnifiedCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]),
     cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]),
     cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("alpha", cam.getParam()[4]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, granite::UnifiedCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 5, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]),
     cereal::make_nvp("cx", intr[2]), cereal::make_nvp("cy", intr[3]),
     cereal::make_nvp("alpha", intr[4]));

  cam = granite::UnifiedCamera<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const granite::PinholeCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]),
     cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]),
     cereal::make_nvp("cy", cam.getParam()[3]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, granite::PinholeCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 4, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]),
     cereal::make_nvp("cx", intr[2]), cereal::make_nvp("cy", intr[3]));

  cam = granite::PinholeCamera<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const granite::DoubleSphereCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]),
     cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]),
     cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("xi", cam.getParam()[4]),
     cereal::make_nvp("alpha", cam.getParam()[5]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, granite::DoubleSphereCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 6, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]),
     cereal::make_nvp("cx", intr[2]), cereal::make_nvp("cy", intr[3]),
     cereal::make_nvp("xi", intr[4]), cereal::make_nvp("alpha", intr[5]));

  cam = granite::DoubleSphereCamera<Scalar>(intr);
}

template <class Archive, class Scalar>
inline void save(Archive& ar, const granite::FovCamera<Scalar>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]),
     cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]),
     cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("w", cam.getParam()[4]));
}

template <class Archive, class Scalar>
inline void load(Archive& ar, granite::FovCamera<Scalar>& cam) {
  Eigen::Matrix<Scalar, 5, 1> intr;

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]),
     cereal::make_nvp("cx", intr[2]), cereal::make_nvp("cy", intr[3]),
     cereal::make_nvp("w", intr[4]));

  cam = granite::FovCamera<Scalar>(intr);
}

template <class Archive, class T>
inline void save(Archive& ar, const granite::ManagedImage<T>& m) {
  ar(m.w);
  ar(m.h);
  ar(cereal::binary_data(m.ptr, sizeof(T) * m.w * m.h));
}

template <class Archive, class T>
inline void load(Archive& ar, granite::ManagedImage<T>& m) {
  size_t w;
  size_t h;
  ar(w);
  ar(h);
  m.Reinitialise(w, h);
  ar(cereal::binary_data(m.ptr, sizeof(T) * m.w * m.h));
}

template <class Archive, class Scalar, int DIM, int ORDER>
inline void save(Archive& ar,
                 const granite::RdSpline<DIM, ORDER, Scalar>& spline) {
  ar(spline.minTimeNs());
  ar(spline.getTimeIntervalNs());
  ar(spline.getKnots());
}

template <class Archive, class Scalar, int DIM, int ORDER>
inline void load(Archive& ar, granite::RdSpline<DIM, ORDER, Scalar>& spline) {
  int64_t start_t_ns;
  int64_t dt_ns;
  Eigen::aligned_deque<Eigen::Matrix<Scalar, DIM, 1>> knots;

  ar(start_t_ns);
  ar(dt_ns);
  ar(knots);

  granite::RdSpline<DIM, ORDER, Scalar> new_spline(dt_ns, start_t_ns);
  for (const auto& k : knots) {
    new_spline.knots_push_back(k);
  }
  spline = new_spline;
}

template <class Archive, class Scalar>
inline void serialize(Archive& ar, granite::Calibration<Scalar>& cam) {
  ar(cereal::make_nvp("T_imu_cam", cam.T_i_c),
     cereal::make_nvp("cam_update_rate", cam.cam_update_rate),
     cereal::make_nvp("intrinsics", cam.intrinsics),
     cereal::make_nvp("resolution", cam.resolution),
     cereal::make_nvp("main_cam_idx", cam.main_cam_idx),
     cereal::make_nvp("stereo_pairs", cam.stereo_pairs),
     cereal::make_nvp("calib_accel_bias", cam.calib_accel_bias.getParam()),
     cereal::make_nvp("calib_gyro_bias", cam.calib_gyro_bias.getParam()),
     cereal::make_nvp("imu_update_rate", cam.imu_update_rate),
     cereal::make_nvp("accel_noise_std", cam.accel_noise_std),
     cereal::make_nvp("gyro_noise_std", cam.gyro_noise_std),
     cereal::make_nvp("accel_bias_std", cam.accel_bias_std),
     cereal::make_nvp("gyro_bias_std", cam.gyro_bias_std),
     cereal::make_nvp("cam_time_offset_ns", cam.cam_time_offset_ns),
     cereal::make_nvp("vignette", cam.vignette));
}

template <class Archive, class Scalar>
inline void serialize(Archive& ar, granite::MocapCalibration<Scalar>& cam) {
  ar(cereal::make_nvp("T_mocap_world", cam.T_moc_w),
     cereal::make_nvp("T_imu_marker", cam.T_i_mark),
     cereal::make_nvp("mocap_time_offset_ns", cam.mocap_time_offset_ns),
     cereal::make_nvp("mocap_to_imu_offset_ns", cam.mocap_to_imu_offset_ns));
}

}  // namespace cereal
