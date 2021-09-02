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

#include <granite/vi_estimator/landmark_database.h>

namespace granite {

void LandmarkDatabase::addLandmark(int lm_id, const KeypointPosition &pos) {
  auto it = kpts.find(lm_id);
  GRANITE_ASSERT(it == kpts.end());

  kpts[lm_id] = pos;
  host_to_kpts[pos.kf_id].emplace(lm_id);
}

void LandmarkDatabase::removeFrame(const FrameId &frame) {
  for (auto it = obs.begin(); it != obs.end(); ++it) {
    for (auto it2 = it->second.cbegin(); it2 != it->second.cend();) {
      if (it2->first.frame_id == frame) {
        for (const auto &v : it2->second) kpts_num_obs.at(v.kpt_id)--;

        it2 = it->second.erase(it2);
      } else {
        ++it2;
      }
    }
  }
}

void LandmarkDatabase::removeKeyframes(
    const std::set<FrameId> &kfs_to_marg,
    const std::set<FrameId> &poses_to_marg,
    const std::set<FrameId> &states_to_marg_all) {
  // remove points
  for (auto it = kpts.cbegin(); it != kpts.cend();) {
    if (kfs_to_marg.count(it->second.kf_id.frame_id) > 0) {
      auto num_obs_it = kpts_num_obs.find(it->first);
      num_observations -= num_obs_it->second;
      kpts_num_obs.erase(num_obs_it);

      it = kpts.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = obs.cbegin(); it != obs.cend();) {
    if (kfs_to_marg.count(it->first.frame_id) > 0) {
      it = obs.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = obs.begin(); it != obs.end(); ++it) {
    for (auto it2 = it->second.cbegin(); it2 != it->second.cend();) {
      if (poses_to_marg.count(it2->first.frame_id) > 0 ||
          states_to_marg_all.count(it2->first.frame_id) > 0 ||
          kfs_to_marg.count(it->first.frame_id) > 0) {
        for (const auto &v : it2->second) kpts_num_obs.at(v.kpt_id)--;

        it2 = it->second.erase(it2);
      } else {
        ++it2;
      }
    }
  }

  for (auto it = host_to_kpts.cbegin(); it != host_to_kpts.cend();) {
    if (kfs_to_marg.count(it->first.frame_id) > 0 ||
        states_to_marg_all.count(it->first.frame_id) > 0 ||
        poses_to_marg.count(it->first.frame_id) > 0) {
      it = host_to_kpts.erase(it);
    } else {
      ++it;
    }
  }
}

std::vector<TimeCamId> LandmarkDatabase::getHostKfs() const {
  std::vector<TimeCamId> res;
  for (const auto &kv : obs) res.emplace_back(kv.first);
  return res;
}

std::vector<KeypointPosition> LandmarkDatabase::getLandmarksForHost(
    const TimeCamId &tcid) const {
  std::vector<KeypointPosition> res;
  for (const auto &v : host_to_kpts.at(tcid)) res.emplace_back(kpts.at(v));
  return res;
}

void LandmarkDatabase::addObservation(const TimeCamId &tcid_target,
                                      const KeypointObservation &o) {
  auto it = kpts.find(o.kpt_id);
  GRANITE_ASSERT(it != kpts.end());

  auto &obs_vec = obs[it->second.kf_id][tcid_target];

  // Check that the point observation is inserted only once
  for (const auto &oo : obs_vec) {
    GRANITE_ASSERT(oo.kpt_id != o.kpt_id);
  }

  obs_vec.emplace_back(o);

  num_observations++;
  kpts_num_obs[o.kpt_id]++;
}

KeypointPosition &LandmarkDatabase::getLandmark(int lm_id) {
  return kpts.at(lm_id);
}

const KeypointPosition &LandmarkDatabase::getLandmark(int lm_id) const {
  return kpts.at(lm_id);
}

std::map<TimeCamId, KeypointObservation>
LandmarkDatabase::getObservationsOfLandmark(const int lm_id) const {
  auto it = kpts.find(lm_id);
  GRANITE_ASSERT(it != kpts.end());

  const TimeCamId &host_id = it->second.kf_id;

  std::map<TimeCamId, KeypointObservation> obs_of_lm;

  for (const auto &kv : obs.at(host_id)) {
    for (const auto &kpt_obs : kv.second) {
      if (kpt_obs.kpt_id == lm_id) {
        obs_of_lm.emplace(std::make_pair(kv.first, kpt_obs));
        break;
      }
    }
  }

  return obs_of_lm;
}

const Eigen::aligned_map<
    TimeCamId,
    Eigen::aligned_map<TimeCamId, Eigen::aligned_vector<KeypointObservation>>>
    &LandmarkDatabase::getObservations() const {
  return obs;
}

Eigen::aligned_vector<KeypointObservation> LandmarkDatabase::getObservations(
    const TimeCamId &tcid) const {
  Eigen::aligned_vector<KeypointObservation> obs_for_tcid;
  for (const auto &kv_host : obs) {
    if (kv_host.second.count(tcid)) {
      obs_for_tcid.insert(obs_for_tcid.end(), kv_host.second.at(tcid).begin(),
                          kv_host.second.at(tcid).end());
    }
  }

  return obs_for_tcid;
}

bool LandmarkDatabase::landmarkExists(int lm_id) const {
  return kpts.count(lm_id) > 0;
}

size_t LandmarkDatabase::numLandmarks() const { return kpts.size(); }

int LandmarkDatabase::numObservations() const { return num_observations; }

int LandmarkDatabase::numObservations(int lm_id) const {
  return kpts_num_obs.at(lm_id);
}

void LandmarkDatabase::removeLandmark(int lm_id) {
  auto it = kpts.find(lm_id);
  GRANITE_ASSERT(it != kpts.end());

  host_to_kpts.at(it->second.kf_id).erase(lm_id);

  std::set<TimeCamId> to_remove;

  for (auto &kv : obs.at(it->second.kf_id)) {
    int idx = -1;
    for (size_t i = 0; i < kv.second.size(); ++i) {
      if (kv.second[i].kpt_id == lm_id) {
        idx = i;
        break;
      }
    }

    if (idx >= 0) {
      GRANITE_ASSERT(kv.second.size() > 0);

      std::swap(kv.second[idx], kv.second[kv.second.size() - 1]);
      kv.second.resize(kv.second.size() - 1);

      num_observations--;
      kpts_num_obs.at(lm_id)--;

      if (kv.second.size() == 0) to_remove.insert(kv.first);
    }
  }

  for (const auto &v : to_remove) {
    obs.at(it->second.kf_id).erase(v);
  }

  GRANITE_ASSERT_STREAM(kpts_num_obs.at(lm_id) == 0, kpts_num_obs.at(lm_id));
  kpts_num_obs.erase(lm_id);
  kpts.erase(lm_id);
}

void LandmarkDatabase::removeObservations(int lm_id,
                                          const std::set<TimeCamId> &outliers) {
  auto it = kpts.find(lm_id);
  GRANITE_ASSERT(it != kpts.end());

  std::set<TimeCamId> to_remove;

  for (auto &kv : obs.at(it->second.kf_id)) {
    if (outliers.count(kv.first) > 0) {
      int idx = -1;
      for (size_t i = 0; i < kv.second.size(); i++) {
        if (kv.second[i].kpt_id == lm_id) {
          idx = i;
          break;
        }
      }
      GRANITE_ASSERT(idx >= 0);
      GRANITE_ASSERT(kv.second.size() > 0);

      std::swap(kv.second[idx], kv.second[kv.second.size() - 1]);
      kv.second.resize(kv.second.size() - 1);

      num_observations--;
      kpts_num_obs.at(lm_id)--;

      if (kv.second.size() == 0) to_remove.insert(kv.first);
    }
  }

  for (const auto &v : to_remove) {
    obs.at(it->second.kf_id).erase(v);
  }
}

}  // namespace granite
