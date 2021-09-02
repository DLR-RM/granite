#!/usr/bin/env python3

import argparse
import subprocess
import csv
import os.path
from send2trash import send2trash
import numpy as np
import shutil
import sophus as sp
from scipy.spatial.transform import Rotation as R

sequences = [
    "room1",
    "room2",
    "room3",
    "room4",
    "room5",
    "room6"
]


def convert_gt(orig_gt_file_name, new_gt_file_name, T_i_c):
    """
    Converts the ground truth poses into the right coordinate frame

    """

    csv_row_dicts = []

    with open(orig_gt_file_name) as old_gt_file:
        old_gt_reader = csv.DictReader(old_gt_file, delimiter=',')
        for row in old_gt_reader:

            T_w_i = sp.SE3(R.from_quat([float(row['qx']), float(row['qy']), float(row['qz']), float(
                row['qw'])]).as_matrix(), [float(row['tx']), float(row['ty']), float(row['tz'])])

            T_w_c = T_w_i * T_i_c
            R_w_c = R.from_matrix(T_w_c.rotationMatrix())

            csv_row_dicts.append({
                "#timestamp [ns]": float(row['# timestamp[ns]']),
                "p_RS_R_x [m]": T_w_c.translation()[0],
                "p_RS_R_y [m]": T_w_c.translation()[1],
                "p_RS_R_z [m]": T_w_c.translation()[2],
                "q_RS_x []": R_w_c.as_quat()[0],
                "q_RS_y []": R_w_c.as_quat()[1],
                "q_RS_z []": R_w_c.as_quat()[2],
                "q_RS_w []": R_w_c.as_quat()[3]
            })

    with open(new_gt_file_name, 'w') as new_gt_file:
        fieldnames = ["#timestamp [ns]", "p_RS_R_x [m]", "p_RS_R_y [m]",
                      "p_RS_R_z [m]", "q_RS_x []", "q_RS_y []", "q_RS_z []", "q_RS_w []"]
        new_gt_writer = csv.DictWriter(new_gt_file, fieldnames=fieldnames)

        new_gt_writer.writeheader()
        for row_dict in csv_row_dicts:
            new_gt_writer.writerow(row_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    This script runs over many sequences (specified in 'sequences') of the TUM-VI dataset. Every sequence is executed 'runs_per_sequence' times. The evaluation script from TUM-RGBD (also used by ORB-SLAM3) is used to calculate the Root Mean Square Absolute Trajectory Error (RMS ATE). The median of all runs is reported in 'rmsate_summary.txt'.
    ''')
    parser.add_argument(
        "--dataset_path",
        help="Path to the TUM-VI dataset. Should lead to a directory that contains the folders '1024_16' and '512_16'.")
    parser.add_argument(
        "--output_path", help="Path to to where the evaluation results should be stored.")
    parser.add_argument(
        "--resolution", help="Either '1024_16' or '512_16'. Default: '512_16'", default="512_16")
    parser.add_argument('--runs_per_sequence',
                        help='How often should every sequence be evaluated. Default: 3', default=3)
    parser.add_argument('--setup',
                        help='Either "mono", "stereo-inertial" or "stereo"', default="stereo-inertial")
    args = parser.parse_args()
    runs_per_sequence = int(args.runs_per_sequence)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    result_file = "%s/trajectory.txt" % dir_path

    summary_file = os.path.join(
        args.output_path, f"{args.setup}_rmsate_summary.txt")

    if os.path.isfile(summary_file):
        print("An old version of 'rmsate_summary.txt' exists. Going to delete it.")
        send2trash(summary_file)

    with open(summary_file, "a") as s_file:
        s_file.write(
            "#sequence name: median RMS ATE, fail count/ runs per sequence\n")

    if args.setup == "mono":
        T_i_c = sp.SE3(R.from_quat([-0.013392900690257393, -0.6945866755293793, 0.7192437840259219,
                                    0.007639340823570553]).as_matrix(), [0.04548094812071685, -0.07145370002838907, -0.046315428444919249])
    else:
        T_i_c = sp.SE3()

    # run over all sequences
    for sequence in sequences:

        sequence_output_path = os.path.join(
            args.output_path, f"{sequence}_{args.setup}")
        if not os.path.isdir(sequence_output_path):
            os.mkdir(sequence_output_path)

        sequence_path = "%s/%s/dataset-%s_%s" % (
            args.dataset_path, args.resolution, sequence, args.resolution)

        print("Looking for a sequence in %s" % sequence_path)

        orig_gt_file_name = os.path.join(sequence_path, "dso/gt_imu.csv")
        new_gt_file_name = os.path.join(dir_path, f"{sequence}-gt.csv")
        convert_gt(orig_gt_file_name, new_gt_file_name, T_i_c)

        # initialize statistics
        rmsates = np.zeros(runs_per_sequence, dtype=np.float64)
        fail_count = 0

        # execute this sequence runs_per_sequence times
        for run_number in range(runs_per_sequence):
            print("Running granite on sequence %s run number %d" %
                  (sequence, run_number + 1))

            failed = False

            # the result.txt file is the indicator if a run was successful
            # we delete it know to see if a new file exists after granite finished
            if os.path.isfile(result_file):
                os.remove(result_file)

            # execute granite
            subprocess.run(["%s/../../build/granite_vio" % dir_path,
                            "--dataset-path", sequence_path,
                            "--cam-calib", os.path.join(
                                dir_path, f"../../data/tumvi_512_ds_calib_{args.setup.replace('-inertial', '')}.json"),
                            "--dataset-type", "euroc",
                            "--config-path", os.path.join(
                                dir_path, "../../data", f"tumvi_512_config_{args.setup.replace('-', '_')}.json"),
                            #"--marg-data", "tumvi_marg_data",
                            "--show-gui", "0",
                            #"--result-path", "room1-result.txt",
                            "--save-trajectory", "tum",
                            "--use-imu", "1" if args.setup == "stereo-inertial" else "0"
                            ],
                           cwd=dir_path)

            # indicator if the run was successful
            if not os.path.isfile(result_file):
                failed = True
                fail_count += 1
                print("Granite on sequence %s run number %d FAILED" %
                      (sequence, run_number + 1))

            if not failed:

                print("Calculating RMS ATE for %s run number %d" %
                      (sequence, run_number + 1))

                # Calculate RMS ATE by using the evaluation script from TUM-RGBD (also used by ORB-SLAM3)
                evaluate_ate_scale_proc = subprocess.Popen(["python2", "-u",
                                                            "%s/evaluate_ate_scale.py" % dir_path,
                                                            new_gt_file_name,
                                                            result_file,
                                                            "--time_factor", "1e9",
                                                            #"--max_difference", "0.1",
                                                            "--plot", "plot-tumvi.svg"],
                                                           cwd=dir_path,
                                                           universal_newlines=True, stdout=subprocess.PIPE,
                                                           stderr=subprocess.STDOUT)

                stdout = evaluate_ate_scale_proc.communicate()[0]
                # parse the output of the evaluation script
                try:
                    if(args.setup == "mono"):
                        rmsate = float(
                            stdout.rstrip().split(',')[2])
                    else:
                        rmsate = float(
                            stdout.rstrip().split(',')[0])
                    if(rmsate > 1000):
                        failed = True
                        fail_count += 1
                        print("Granite on sequence %s run number %d FAILED" %
                              (sequence, run_number + 1))
                    else:
                        rmsates[run_number] = rmsate
                        print("RMS ATE: %f" % rmsates[run_number])
                except:
                    print(stdout)

                if os.path.isfile(result_file):
                    shutil.move(result_file, os.path.join(
                        sequence_output_path, f"{run_number + 1}_trajectory.txt"))
                if os.path.isfile("plot-tumvi.svg"):
                    shutil.move("plot-tumvi.svg", os.path.join(
                        sequence_output_path, f"{run_number + 1}_plot.svg"))

        # get median of runs
        median = np.NaN
        if fail_count < runs_per_sequence:
            median = np.median(rmsates[rmsates != 0])

        # write statistics
        with open(summary_file, "a") as s_file:
            s_file.write("%s: %f, %d/%d\n" %
                         (sequence, median, fail_count, runs_per_sequence))

        print("median RMS ATE of %s: %f" % (sequence, median))
        print("failed %d/%d" % (fail_count, runs_per_sequence))
