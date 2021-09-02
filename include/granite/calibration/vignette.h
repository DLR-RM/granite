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

#include <granite/calibration/aprilgrid.h>
#include <granite/calibration/calibration_helper.h>

#include <granite/spline/rd_spline.h>

namespace granite {

class VignetteEstimator {
 public:
  static const int SPLINE_N = 4;
  static const int64_t knot_spacing = 1e10;
  static const int border_size = 2;

  VignetteEstimator(
      const VioDatasetPtr &vio_dataset,
      const Eigen::aligned_vector<Eigen::Vector2d> &optical_centers,
      const Eigen::aligned_vector<Eigen::Vector2i> &resolutions,
      const std::map<TimeCamId, Eigen::aligned_vector<Eigen::Vector3d>>
          &reprojected_vignette,
      const AprilGrid &april_grid);

  void compute_error(std::map<TimeCamId, std::vector<double>>
                         *reprojected_vignette_error = nullptr);

  void opt_irradience();

  void opt_vign();

  void optimize();

  void compute_data_log(std::vector<std::vector<float>> &vign_data_log);

  void save_vign_png(const std::string &path);

  inline const std::vector<granite::RdSpline<1, SPLINE_N>> &get_vign_param() {
    return vign_param;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  const VioDatasetPtr vio_dataset;
  Eigen::aligned_vector<Eigen::Vector2d> optical_centers;
  Eigen::aligned_vector<Eigen::Vector2i> resolutions;
  std::map<TimeCamId, Eigen::aligned_vector<Eigen::Vector3d>>
      reprojected_vignette;
  const AprilGrid &april_grid;

  size_t vign_size;
  std::vector<double> irradiance;
  std::vector<granite::RdSpline<1, SPLINE_N>> vign_param;
};
}  // namespace granite
