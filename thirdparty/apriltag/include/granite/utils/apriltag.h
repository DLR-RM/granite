
#include <granite/image/image.h>
#include <granite/utils/sophus_utils.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>

namespace granite {

struct ApriltagDetectorData;

class ApriltagDetector {
 public:
  ApriltagDetector();

  ~ApriltagDetector();

  void detectTags(const granite::ManagedImage<uint16_t>& img_raw,
                  Eigen::aligned_vector<Eigen::Vector2d>& corners,
                  std::vector<int>& ids, std::vector<double>& radii,
                  Eigen::aligned_vector<Eigen::Vector2d>& corners_rejected,
                  std::vector<int>& ids_rejected,
                  std::vector<double>& radii_rejected);

  void detectTags(const granite::ManagedImage<uint8_t>& img_raw,
                  Eigen::aligned_vector<Eigen::Vector2d>& corners,
                  std::vector<int>& ids, std::vector<double>& radii,
                  Eigen::aligned_vector<Eigen::Vector2d>& corners_rejected,
                  std::vector<int>& ids_rejected,
                  std::vector<double>& radii_rejected);

  void detectTags(const cv::Mat& image,
                  Eigen::aligned_vector<Eigen::Vector2d>& corners,
                  std::vector<int>& ids, std::vector<double>& radii,
                  Eigen::aligned_vector<Eigen::Vector2d>& corners_rejected,
                  std::vector<int>& ids_rejected,
                  std::vector<double>& radii_rejected);

 private:
  ApriltagDetectorData* data;
};

}  // namespace granite
