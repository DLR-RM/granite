#include <benchmark/benchmark.h>

#include <granite/camera/generic_camera.hpp>

template <class CamT>
void BM_Project(benchmark::State &state) {
  static const int SIZE = 50;

  typedef typename CamT::Vec4 Vec4;
  typedef typename CamT::Vec2 Vec2;

  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  Vec4 p(0, 0, 5, 1);

  for (auto _ : state) {
    for (const CamT &cam : test_cams) {
      for (int x = -SIZE; x < SIZE; x++) {
        for (int y = -SIZE; y < SIZE; y++) {
          p[0] = x;
          p[1] = y;

          Vec2 res;
          benchmark::DoNotOptimize(cam.project(p, res));
        }
      }
    }
  }
}

template <class CamT>
void BM_ProjectJacobians(benchmark::State &state) {
  static const int SIZE = 50;

  typedef typename CamT::Vec2 Vec2;
  typedef typename CamT::Vec4 Vec4;

  typedef typename CamT::Mat24 Mat24;
  typedef typename CamT::Mat2N Mat2N;

  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  Mat24 Jp;
  Mat2N Jparam;

  Vec4 p(0, 0, 5, 1);

  for (auto _ : state) {
    for (const CamT &cam : test_cams) {
      for (int x = -SIZE; x <= SIZE; x++) {
        for (int y = -SIZE; y <= SIZE; y++) {
          p[0] = x;
          p[1] = y;

          Vec2 res;
          benchmark::DoNotOptimize(cam.project(p, res, &Jp, &Jparam));
        }
      }
    }
  }
}

template <class CamT>
void BM_Unproject(benchmark::State &state) {
  static const int SIZE = 50;

  typedef typename CamT::Vec2 Vec2;
  typedef typename CamT::Vec4 Vec4;

  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  for (auto _ : state) {
    for (const CamT &cam : test_cams) {
      Vec2 p_center(cam.getParam()(2), cam.getParam()(3));

      for (int x = -SIZE; x <= SIZE; x++) {
        for (int y = -SIZE; y <= SIZE; y++) {
          Vec2 p = p_center;
          p[0] += x;
          p[1] += y;

          Vec4 res;
          benchmark::DoNotOptimize(cam.unproject(p, res));
        }
      }
    }
  }
}

template <class CamT>
void BM_UnprojectJacobians(benchmark::State &state) {
  static const int SIZE = 50;

  typedef typename CamT::Vec2 Vec2;
  typedef typename CamT::Vec4 Vec4;

  Eigen::aligned_vector<CamT> test_cams = CamT::getTestProjections();

  typedef typename CamT::Mat42 Mat42;
  typedef typename CamT::Mat4N Mat4N;

  Mat42 Jp;
  Mat4N Jparam;

  for (auto _ : state) {
    for (const CamT &cam : test_cams) {
      Vec2 p_center(cam.getParam()(2), cam.getParam()(3));

      for (int x = -SIZE; x <= SIZE; x++) {
        for (int y = -SIZE; y <= SIZE; y++) {
          Vec2 p = p_center;
          p[0] += x;
          p[1] += y;

          Vec4 res;

          benchmark::DoNotOptimize(cam.unproject(p, res, &Jp, &Jparam));
        }
      }
    }
  }
}

BENCHMARK_TEMPLATE(BM_Project, granite::PinholeCamera<double>);
BENCHMARK_TEMPLATE(BM_Project, granite::ExtendedUnifiedCamera<double>);
BENCHMARK_TEMPLATE(BM_Project, granite::UnifiedCamera<double>);
BENCHMARK_TEMPLATE(BM_Project, granite::KannalaBrandtCamera4<double>);
BENCHMARK_TEMPLATE(BM_Project, granite::DoubleSphereCamera<double>);
BENCHMARK_TEMPLATE(BM_Project, granite::FovCamera<double>);

BENCHMARK_TEMPLATE(BM_ProjectJacobians, granite::PinholeCamera<double>);
BENCHMARK_TEMPLATE(BM_ProjectJacobians, granite::ExtendedUnifiedCamera<double>);
BENCHMARK_TEMPLATE(BM_ProjectJacobians, granite::UnifiedCamera<double>);
BENCHMARK_TEMPLATE(BM_ProjectJacobians, granite::KannalaBrandtCamera4<double>);
BENCHMARK_TEMPLATE(BM_ProjectJacobians, granite::DoubleSphereCamera<double>);
BENCHMARK_TEMPLATE(BM_ProjectJacobians, granite::FovCamera<double>);

BENCHMARK_TEMPLATE(BM_Unproject, granite::PinholeCamera<double>);
BENCHMARK_TEMPLATE(BM_Unproject, granite::ExtendedUnifiedCamera<double>);
BENCHMARK_TEMPLATE(BM_Unproject, granite::UnifiedCamera<double>);
BENCHMARK_TEMPLATE(BM_Unproject, granite::KannalaBrandtCamera4<double>);
BENCHMARK_TEMPLATE(BM_Unproject, granite::DoubleSphereCamera<double>);
BENCHMARK_TEMPLATE(BM_Unproject, granite::FovCamera<double>);

BENCHMARK_TEMPLATE(BM_UnprojectJacobians, granite::PinholeCamera<double>);
BENCHMARK_TEMPLATE(BM_UnprojectJacobians,
                   granite::ExtendedUnifiedCamera<double>);
BENCHMARK_TEMPLATE(BM_UnprojectJacobians, granite::UnifiedCamera<double>);
BENCHMARK_TEMPLATE(BM_UnprojectJacobians, granite::KannalaBrandtCamera4<double>);
BENCHMARK_TEMPLATE(BM_UnprojectJacobians, granite::DoubleSphereCamera<double>);
BENCHMARK_TEMPLATE(BM_UnprojectJacobians, granite::FovCamera<double>);

BENCHMARK_MAIN();
