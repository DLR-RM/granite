cmake_minimum_required(VERSION 3.10)

# Note: add_subdirectory(googletest ...) is called in granite-headers

include_directories(../thirdparty/granite-headers/test/include)


add_executable(test_spline_opt src/test_spline_opt.cpp)
target_link_libraries(test_spline_opt gtest gtest_main granite)

add_executable(test_vio src/test_vio.cpp)
target_link_libraries(test_vio gtest gtest_main granite)

add_executable(test_nfr src/test_nfr.cpp)
target_link_libraries(test_nfr gtest gtest_main granite)

add_executable(test_mono_map_initialization src/test_mono_map_initialization.cpp)
target_link_libraries(test_mono_map_initialization gtest gtest_main granite opengv)

add_executable(test_rel_pose_constraint src/test_rel_pose_constraint.cpp)
target_link_libraries(test_rel_pose_constraint gtest gtest_main granite )


enable_testing()

include(GoogleTest)

#gtest_discover_tests(test_spline_opt DISCOVERY_TIMEOUT 60)
#gtest_discover_tests(test_vio DISCOVERY_TIMEOUT 60)
#gtest_discover_tests(test_nfr DISCOVERY_TIMEOUT 60)

gtest_add_tests(TARGET test_spline_opt AUTO)
gtest_add_tests(TARGET test_vio AUTO)
gtest_add_tests(TARGET test_nfr AUTO)
gtest_add_tests(TARGET test_mono_map_initialization AUTO)
gtest_add_tests(TARGET test_rel_pose_constraint AUTO)
