#!/usr/bin/env python3

import os
import argparse
import glob
import sophus as sp
import numpy as np
from scipy.spatial.transform import Rotation as R

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert morocco sequence to EuRoC format.")
    parser.add_argument("--dataset_path", help="Path to the sequence")

    args = parser.parse_args()

    if not os.path.isdir(args.dataset_path):
        print("Dataset directory does not exist")
        exit

    sequence_name = os.path.basename(
        os.path.realpath(args.dataset_path)).replace("/", "")

    print("sequence_name: ", sequence_name)

    timestamps_mono_path = os.path.join(args.dataset_path, "timestamps-mono.txt")
    timestamps_stereo_path = os.path.join(args.dataset_path, "timestamps-stereo.txt")
    times_path = os.path.join(args.dataset_path, "times.txt")

    left_images_path = os.path.join(args.dataset_path, "rect_left")
    right_images_path = os.path.join(args.dataset_path, "rect_right")

    timestamps_left = []
    timestamps_right = []
    timestamps = []

    for img_left_path in glob.glob(os.path.join(left_images_path, "*.png")):
        img_left_name = os.path.basename(img_left_path)
        img_left_name_split = img_left_name.split("_")
        if len(img_left_name_split) >= 4:
            timestamp = img_left_name_split[3].replace(".png", "")
        else:
            timestamp = img_left_name.replace(".png", "")
        os.rename(img_left_path, os.path.join(
            left_images_path, timestamp + ".png"))

        timestamps_left.append(int(timestamp))

    for img_right_path in glob.glob(os.path.join(right_images_path, "*.png")):
        img_right_name = os.path.basename(img_right_path)
        img_right_name_split = img_right_name.split("_")
        if len(img_right_name_split) >= 4:
            timestamp = img_right_name_split[3].replace(".png", "")
        else:
            timestamp = img_right_name.replace(".png", "")

        os.rename(img_right_path, os.path.join(
            right_images_path, timestamp + ".png"))

        timestamps_right.append(int(timestamp))

        if int(timestamp) in timestamps_left:
            timestamps.append(int(timestamp))

    timestamps.sort()
    timestamps_left.sort()
    timestamps_right.sort()

    # write timestamps file for ORB-SLAM
    with open(timestamps_mono_path, 'w') as timestamps_file:
        for t_ns in timestamps_left:
            timestamps_file.write(str(t_ns))
            timestamps_file.write("\n")

    with open(timestamps_stereo_path, 'w') as timestamps_file:
        for t_ns in timestamps:
            timestamps_file.write(str(t_ns))
            timestamps_file.write("\n")

    # write times file for DSO 
    with open(times_path, 'w') as timestamps_file:
        for t_ns in timestamps_left:
            timestamps_file.write(str(t_ns))
            timestamps_file.write(" ")
            timestamps_file.write(f"{str(t_ns)[0:-9]}.{str(t_ns)[-9:]}")
            timestamps_file.write("\n")

    mav0_path = os.path.join(args.dataset_path, "mav0")
    cam0_path = os.path.join(mav0_path, "cam0")
    cam0_data_path = os.path.join(cam0_path, "data.csv")
    cam0_data_folder_path = os.path.join(cam0_path, "data")
    cam1_path = os.path.join(mav0_path, "cam1")
    cam1_data_path = os.path.join(cam1_path, "data.csv")
    cam1_data_folder_path = os.path.join(cam1_path, "data")
    imu0_path = os.path.join(mav0_path, "imu0")
    imu0_data_path = os.path.join(imu0_path, "data.csv")
    gt_path = os.path.join(mav0_path, "gt")
    gt_data_path = os.path.join(gt_path, "data.csv")

    if not os.path.isdir(mav0_path):
        os.mkdir(mav0_path)
    if not os.path.isdir(cam0_path):
        os.mkdir(cam0_path)
    if not os.path.isdir(cam0_data_folder_path):
        os.symlink(os.path.relpath(left_images_path,
                                   cam0_path), cam0_data_folder_path)

    if not os.path.isdir(cam1_path):
        os.mkdir(cam1_path)
    if not os.path.isdir(imu0_path):
        os.mkdir(imu0_path)
    if not os.path.isdir(cam1_data_folder_path):
        os.symlink(os.path.relpath(right_images_path,
                                   cam1_path), cam1_data_folder_path)
    if not os.path.isdir(gt_path):
        os.mkdir(gt_path)

    with open(imu0_data_path, 'w') as imu0_data_file:
        with open(os.path.join(args.dataset_path, sequence_name + "_imu_data.csv")) as imu_ros_file:
            for row in imu_ros_file:
                if(row[0] != "%"):
                    cells = row.split(",")
                    imu0_data_file.write(str(int(cells[0])))
                    imu0_data_file.write(",")

                    imu0_data_file.write(str(float(cells[4])))
                    imu0_data_file.write(",")
                    imu0_data_file.write(str(float(cells[5])))
                    imu0_data_file.write(",")
                    imu0_data_file.write(str(float(cells[6])))
                    imu0_data_file.write(",")
                    imu0_data_file.write(str(float(cells[7])))
                    imu0_data_file.write(",")
                    imu0_data_file.write(str(float(cells[8])))
                    imu0_data_file.write(",")
                    imu0_data_file.write(str(float(cells[9])))
                    imu0_data_file.write("\n")

    with open(cam0_data_path, 'w') as data_file:
        for t_ns in timestamps_left:
            data_file.write(f"{t_ns},{t_ns}.png")
            data_file.write("\n")

    with open(cam1_data_path, 'w') as data_file:
        for t_ns in timestamps_right:
            data_file.write(f"{t_ns},{t_ns}.png")
            data_file.write("\n")

    with open(os.path.join(args.dataset_path, "ground_truth", "gt_5DoF_gnss.csv")) as gt_source_file:
        with open(gt_data_path, 'w') as gt_data_file:
            gt_data_file.write("#timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]\n")

            T_imu_B = sp.SE3(np.eye(3), [0.07, 0, 0.13])

            for row in gt_source_file:
                if(row[0] != '%'):
                    cells = row.split(',')

                    T_w_imu = sp.SE3(R.from_quat([float(cells[10]), float(cells[11]), float(cells[12]), float(
                        cells[9])]).as_matrix(), [float(cells[3]), float(cells[4]), float(cells[5])]) * T_imu_B

                    gt_data_file.write(str(int(float(cells[0]) * 1e9)))
                    gt_data_file.write(",")

                    gt_data_file.write(str(T_w_imu.translation()[0]))
                    gt_data_file.write(",")
                    gt_data_file.write(str(T_w_imu.translation()[1]))
                    gt_data_file.write(",")
                    gt_data_file.write(str(T_w_imu.translation()[2]))
                    gt_data_file.write(",")

                    q_w_imu = R.from_matrix(T_w_imu.rotationMatrix()).as_quat()

                    gt_data_file.write(str(q_w_imu[3]))
                    gt_data_file.write(",")
                    gt_data_file.write(str(q_w_imu[0]))
                    gt_data_file.write(",")
                    gt_data_file.write(str(q_w_imu[1]))
                    gt_data_file.write(",")
                    gt_data_file.write(str(q_w_imu[2]))
                    gt_data_file.write(",")

                    gt_data_file.write(str(float(cells[16])))
                    gt_data_file.write(",")
                    gt_data_file.write(str(float(cells[17])))
                    gt_data_file.write(",")
                    gt_data_file.write(str(float(cells[18])))

                    for i in range(6):
                        gt_data_file.write(",")
                        gt_data_file.write("0.0")

                    gt_data_file.write("\n")
