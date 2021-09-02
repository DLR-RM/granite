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

#include <granite/utils/imu_types.h>

namespace granite {

// keypoint position defined relative to some frame
struct KeypointPosition {
  TimeCamId kf_id;
  Eigen::Vector2d dir;
  double id;
  double weight;

  inline void backup() {
    backup_dir = dir;
    backup_id = id;
  }

  inline void restore() {
    dir = backup_dir;
    id = backup_id;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  Eigen::Vector2d backup_dir;
  double backup_id;
};

struct KeypointObservation {
  int kpt_id;
  Eigen::Vector2d pos;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class LandmarkDatabase {
 public:
  // Non-const
  void addLandmark(int lm_id, const KeypointPosition& pos);

  void removeFrame(const FrameId& frame);

  void removeKeyframes(const std::set<FrameId>& kfs_to_marg,
                       const std::set<FrameId>& poses_to_marg,
                       const std::set<FrameId>& states_to_marg_all);

  void addObservation(const TimeCamId& tcid_target,
                      const KeypointObservation& o);

  KeypointPosition& getLandmark(int lm_id);

  // Const
  const KeypointPosition& getLandmark(int lm_id) const;

  std::vector<TimeCamId> getHostKfs() const;

  std::vector<KeypointPosition> getLandmarksForHost(
      const TimeCamId& tcid) const;

  std::map<TimeCamId, KeypointObservation> getObservationsOfLandmark(
      int lm_id) const;

  const Eigen::aligned_map<
      TimeCamId, Eigen::aligned_map<
                     TimeCamId, Eigen::aligned_vector<KeypointObservation>>>&
  getObservations() const;

  Eigen::aligned_vector<KeypointObservation> getObservations(
      const TimeCamId& tcid) const;

  bool landmarkExists(int lm_id) const;

  size_t numLandmarks() const;

  int numObservations() const;

  int numObservations(int lm_id) const;

  void removeLandmark(int lm_id);

  void removeObservations(int lm_id, const std::set<TimeCamId>& obs);

  inline void backup() {
    for (auto& kv : kpts) kv.second.backup();
  }

  inline void restore() {
    for (auto& kv : kpts) kv.second.restore();
  }

  inline void clear() {
    kpts.clear();
    obs.clear();
    host_to_kpts.clear();
    num_observations = 0;
    kpts_num_obs.clear();
  }

 private:
  Eigen::aligned_unordered_map<int, KeypointPosition> kpts;
  Eigen::aligned_map<
      TimeCamId,
      Eigen::aligned_map<TimeCamId, Eigen::aligned_vector<KeypointObservation>>>
      obs;

  std::unordered_map<TimeCamId, std::set<int>> host_to_kpts;

  int num_observations = 0;
  Eigen::aligned_unordered_map<int, int> kpts_num_obs;
};

}  // namespace granite
