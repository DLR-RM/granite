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

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
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

#include <bitset>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>

#include <tbb/concurrent_unordered_map.h>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <sophus/se3.hpp>

#include <cereal/cereal.hpp>

#include <granite/utils/hash.h>
#include <granite/utils/sophus_utils.hpp>

namespace granite {

/// data precision of images
using PixelType = uint8_t;
// using PixelType = uint16_t;

/// ids for 2D features detected in images
using FeatureId = int;

/// identifies a frame of multiple images (stereo pair)
using FrameId = int64_t;

/// identifies the camera (left or right)
using CamId = std::size_t;

/// pair of image timestamp and camera id identifies an image (imageId)
struct TimeCamId {
  TimeCamId() : frame_id(0), cam_id(0) {}

  TimeCamId(const FrameId& frame_id, const CamId& cam_id)
      : frame_id(frame_id), cam_id(cam_id) {}

  FrameId frame_id;
  CamId cam_id;
};

inline std::ostream& operator<<(std::ostream& os, const TimeCamId& tcid) {
  os << tcid.frame_id << "_" << tcid.cam_id;
  return os;
}

inline bool operator<(const TimeCamId& o1, const TimeCamId& o2) {
  if (o1.frame_id == o2.frame_id) return o1.cam_id < o2.cam_id;
  return o1.frame_id < o2.frame_id;
}

inline bool operator==(const TimeCamId& o1, const TimeCamId& o2) {
  return o1.frame_id == o2.frame_id && o1.cam_id == o2.cam_id;
}

inline bool operator!=(const TimeCamId& o1, const TimeCamId& o2) {
  return o1.frame_id != o2.frame_id || o1.cam_id != o2.cam_id;
}

///  Pair of two frames
/// Can be used to index relative pose constraints
struct FramePair {
  FramePair() : frame_first(0), frame_second(0) {}

  FramePair(const FrameId& frame_a, const FrameId& frame_b) {
    if (frame_a < frame_b) {
      frame_first = frame_a;
      frame_second = frame_b;
    } else {
      frame_first = frame_b;
      frame_second = frame_a;
    }
  }

  inline bool contains(const FrameId other) const {
    return frame_first == other || frame_second == other;
  }

  FrameId frame_first, frame_second;
};

inline std::ostream& operator<<(std::ostream& os, const FramePair& fp) {
  os << fp.frame_first << "->" << fp.frame_second;
  return os;
}

inline bool operator<(const FramePair& o1, const FramePair& o2) {
  if (o1.frame_first == o2.frame_first)
    return o1.frame_second < o2.frame_second;
  return o1.frame_first < o2.frame_first;
}

inline bool operator==(const FramePair& o1, const FramePair& o2) {
  return o1.frame_first == o2.frame_first && o1.frame_second == o2.frame_second;
}

inline bool operator!=(const FramePair& o1, const FramePair& o2) {
  return o1.frame_first != o2.frame_first || o1.frame_second != o2.frame_second;
}

constexpr static const size_t FEATURE_HASH_MAX_SIZE = 32;
using FeatureHash = std::bitset<FEATURE_HASH_MAX_SIZE>;
using HashBowVector = std::vector<std::pair<FeatureHash, double>>;

/// keypoint positions and descriptors for an image
struct KeypointsData {
  /// collection of 2d corner points (indexed by FeatureId)
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      corners;
  /// collection of feature orientation (in radian) with same index as `corners`
  /// (indexed by FeatureId)
  std::vector<double> corner_angles;
  /// collection of feature descriptors with same index as `corners` (indexed by
  /// FeatureId)
  std::vector<std::bitset<256>> corner_descriptors;

  Eigen::aligned_vector<Eigen::Vector4d> corners_3d;

  std::vector<FeatureHash> hashes;
  HashBowVector bow_vector;
};

/// feature corners is a collection of { imageId => KeypointsData }
using Corners = tbb::concurrent_unordered_map<TimeCamId, KeypointsData,
                                              std::hash<TimeCamId>>;

/// feature matches for an image pair
struct MatchData {
  /// estimated transformation (based on inliers or calibration) from the second
  /// image's coordinate system to the first image's corrdinate system
  Sophus::SE3d T_i_j;
  /// collection of {featureId_i, featureId_j} pairs of all matches
  std::vector<std::pair<FeatureId, FeatureId>> matches;
  /// collection of {featureId_i, featureId_j} pairs of inlier matches
  std::vector<std::pair<FeatureId, FeatureId>> inliers;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// feature matches is a collection of { (imageId, imageId) => MatchData }
using Matches = tbb::concurrent_unordered_map<
    std::pair<TimeCamId, TimeCamId>, MatchData,
    std::hash<std::pair<TimeCamId, TimeCamId>>,
    std::equal_to<std::pair<TimeCamId, TimeCamId>>,
    Eigen::aligned_allocator<
        std::pair<const std::pair<TimeCamId, TimeCamId>, MatchData>>>;

/// pair of image and feature indices
using ImageFeaturePair = std::pair<TimeCamId, FeatureId>;

/// Feature tracks are collections of {ImageId => FeatureId}.
/// I.e. a collection of all images that observed this feature and the
/// corresponding feature index in that image.
using FeatureTrack = std::map<TimeCamId, FeatureId>;

/// Ids for feature tracks; also used for landmarks created from (some of) the
/// tracks;
using TrackId = int64_t;

/// FeatureTracks is a collection {TrackId => FeatureTrack}
using FeatureTracks = std::unordered_map<TrackId, FeatureTrack>;

/// cameras in the map
struct Camera {
  // camera pose (transforms from camera to world)
  Sophus::SE3d T_w_c;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// landmarks in the map
struct Landmark {
  /// 3d position in world coordinates
  Eigen::Vector3d p;

  /// Inlier observations in the current map.
  /// This is a subset of the original feature track.
  FeatureTrack obs;

  /// Outlier observations in the current map.
  /// This is a subset of the original feature track.
  FeatureTrack outlier_obs;
};

/// collection {imageId => Camera} for all cameras in the map
using Cameras =
    std::map<TimeCamId, Camera, std::less<TimeCamId>,
             Eigen::aligned_allocator<std::pair<const TimeCamId, Camera>>>;

/// collection {trackId => Landmark} for all landmarks in the map.
/// trackIds correspond to feature_tracks
using Landmarks = std::unordered_map<TrackId, Landmark>;

/// camera candidate to be added to map
struct CameraCandidate {
  TimeCamId tcid;
  std::vector<TrackId> shared_tracks;

  // keep track of different stages of adding a set of candidate cameras and its
  // landmarks to the map
  bool tried = false;            //!< tried to add to map
  bool camera_added = false;     //!< succeeded to add to map
  bool landmarks_added = false;  //!< added new landmarks to map
};

/// list of current candidates and some book keeping for the different stages
struct CameraCandidates {
  enum Stage {
    ComputeCandidates,
    AddCameras,
    AddLandmarks,
    Optimize,
    RemoveOutliers,
    Done
  };

  std::vector<CameraCandidate> cameras;
  Stage current_stage = ComputeCandidates;
  int min_localization_inliers = 0;
  int max_cameras_to_add = 0;

  int num_cameras_added() {
    int num_added = 0;
    for (const auto& c : cameras) {
      if (c.camera_added) {
        ++num_added;
      }
    }
    return num_added;
  }

  int num_landmarks_added() {
    int num_added = 0;
    for (const auto& c : cameras) {
      if (c.landmarks_added) {
        ++num_added;
      }
    }
    return num_added;
  }
};

/// Flags for different landmark outlier criteria
enum OutlierFlags {
  OutlierNone = 0,
  // reprojection error much too large
  OutlierReprojectionErrorHuge = 1 << 0,
  // reprojection error too large
  OutlierReprojectionErrorNormal = 1 << 1,
  // distance to a camera too small
  OutlierCameraDistance = 1 << 2,
  // z-coord in some camera frame too small
  OutlierZCoordinate = 1 << 3
};

/// info on a single projected landmark
struct ProjectedLandmark {
  Eigen::Vector2d point_measured;            //!< detected feature location
  Eigen::Vector2d point_reprojected;         //!< landmark projected into image
  Eigen::Vector3d point_3d_c;                //!< 3d point in camera coordinates
  TrackId track_id = -1;                     //!< corresponding track_id
  double reprojection_error = 0;             //!< current reprojection error
  unsigned int outlier_flags = OutlierNone;  //!< flags for outlier
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using ProjectedLandmarkPtr = std::shared_ptr<ProjectedLandmark>;
using ProjectedLandmarkConstPtr = std::shared_ptr<const ProjectedLandmark>;

/// all landmark projections for inlier and outlier observations for a single
/// image
struct ImageProjection {
  std::vector<ProjectedLandmarkConstPtr> obs;
  std::vector<ProjectedLandmarkConstPtr> outlier_obs;
};

/// projections for all images
using ImageProjections = std::map<TimeCamId, ImageProjection>;

/// inlier projections indexed per track
using TrackProjections =
    std::unordered_map<TrackId, std::map<TimeCamId, ProjectedLandmarkConstPtr>>;

}  // namespace granite

namespace cereal {

template <class Archive>
void serialize(Archive& ar, granite::TimeCamId& c) {
  ar(c.frame_id, c.cam_id);
}

template <class Archive>
void serialize(Archive& ar, granite::KeypointsData& c) {
  ar(c.corners, c.corner_angles, c.corner_descriptors);
}

template <class Archive>
void serialize(Archive& ar, granite::MatchData& c) {
  ar(c.T_i_j, c.matches, c.inliers);
}

}  // namespace cereal

namespace std {

template <>
struct hash<granite::TimeCamId> {
  size_t operator()(const granite::TimeCamId& x) const {
    size_t seed = 0;
    granite::hash_combine(seed, x.frame_id);
    granite::hash_combine(seed, x.cam_id);
    return seed;
  }
};

template <>
struct hash<std::pair<granite::TimeCamId, granite::TimeCamId>> {
  size_t operator()(
      const std::pair<granite::TimeCamId, granite::TimeCamId>& x) const {
    size_t seed = 0;
    granite::hash_combine(seed, x.first.frame_id);
    granite::hash_combine(seed, x.first.cam_id);
    granite::hash_combine(seed, x.second.frame_id);
    granite::hash_combine(seed, x.second.cam_id);
    return seed;
  }
};

template <>
struct hash<granite::FramePair> {
  size_t operator()(const granite::FramePair& x) const {
    size_t seed = 0;
    granite::hash_combine(seed, x.frame_first);
    granite::hash_combine(seed, x.frame_second);
    return seed;
  }
};

template <>
struct hash<std::pair<granite::FramePair, granite::FramePair>> {
  size_t operator()(
      const std::pair<granite::FramePair, granite::FramePair>& x) const {
    size_t seed = 0;
    granite::hash_combine(seed, x.first.frame_first);
    granite::hash_combine(seed, x.first.frame_second);
    granite::hash_combine(seed, x.second.frame_first);
    granite::hash_combine(seed, x.second.frame_second);
    return seed;
  }
};

}  // namespace std
