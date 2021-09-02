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

def convert_gt(orig_gt_file_name, new_gt_file_name, T_i_c):
    """
    Converts the ground truth poses into the right coordinate frame

    """

    csv_row_dicts = []

    with open(orig_gt_file_name) as old_gt_file:
        for row in old_gt_file:
            if row[0] == '#':
                continue
            
            cells = row.split(",")
            T_w_i = sp.SE3(R.from_quat([float(cells[5]), float(cells[6]), float(cells[7]), float(cells[4])]).as_matrix(), [float(cells[1]), float(cells[2]), float(cells[3])])

            T_w_c = T_w_i * T_i_c
            R_w_c = R.from_matrix(T_w_c.rotationMatrix())

            csv_row_dicts.append({
                "#timestamp [ns]": float(cells[0]),
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

sequences = [
    "MH01",
    "MH02",
    "MH03",
    "MH04",
    "MH05",
    "V101",
    "V102",
    "V103",
    "V201",
    "V202",
    "V203"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    This script runs over many sequences (specified in 'sequences') of the euRoC dataset. Every sequence is executed 'runs_per_sequence' times. The evaluation script from TUM-RGBD (also used by ORB-SLAM3) is used to calculate the Root Mean Square Absolute Trajectory Error (RMS ATE). The median of all runs is reported in 'rmsate_summary.txt'.
    ''')
    parser.add_argument("--dataset_path", help="Path to the EuRoC dataset.")
    parser.add_argument(
        "--output_path", help="Path to to where the evaluation results should be stored.")
    parser.add_argument('--runs_per_sequence', type=int,
                        help='How often should every sequence be evaluated. Default: 3', default=3)
    parser.add_argument('--setup',
                        help='Either "mono", "stereo-inertial" or "stereo"', default="stereo-inertial")
    args = parser.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    result_file = os.path.join(dir_path, "trajectory.txt")

    summary_file = os.path.join(
        args.output_path, f"{args.setup}_rmsate_summary.txt")

    if os.path.isfile(summary_file):
        print("An old version of 'rmsate_summary.txt' exists. Going to trash it.")
        send2trash(summary_file)

    with open(summary_file, "a") as s_file:
        s_file.write(
            "#sequence name: median RMS ATE, fail count/ runs per sequence, percent tracked\n")

    if args.setup == "mono":
        T_i_c = sp.SE3(R.from_quat([-0.007239825785317818, 0.007541278561558601, 0.7017845426564943,
                                    0.7123125505904486]).as_matrix(), [-0.016774788924641534, -0.068938940687127, 0.005139123188382424])
    else:
        T_i_c = sp.SE3()

    # run over all sequences
    for sequence in sequences:

        sequence_output_path = os.path.join(
            args.output_path, f"{sequence}_{args.setup}")
        if not os.path.isdir(sequence_output_path):
            os.mkdir(sequence_output_path)

        sequence_path = os.path.join(args.dataset_path, sequence)

        print("Looking for a sequence in %s" % sequence_path)

        orig_gt_file_name = os.path.join(sequence_path, "mav0", "state_groundtruth_estimate0", "data.csv")
        new_gt_file_name = os.path.join(dir_path, f"{sequence}-gt.csv")
        convert_gt(orig_gt_file_name, new_gt_file_name, T_i_c)

        # initialize statistics
        rmsates = np.zeros(args.runs_per_sequence, dtype=np.float64)
        percents_tracked = np.zeros(args.runs_per_sequence, dtype=np.float64)
        fail_count = 0

        with open(os.path.join(sequence_path, "mav0", "cam0", "data.csv")) as timestamps_file:
            lines = timestamps_file.readlines()
            first_frame_t = int(lines[1].split(",")[0])
            last_frame_t = int(lines[-1].split(",")[0])

        # execute this sequence runs_per_sequence times
        for run_number in range(args.runs_per_sequence):
            print(
                f"Running granite on sequence {sequence} run number {run_number + 1}")

            failed = False

            # the result.txt file is the indicator if a run was successful
            # we delete it know to see if a new file exists after granite finished
            if os.path.isfile(result_file):
                os.remove(result_file)

            # execute granite
            subprocess.run(["%s/../../build/granite_vio" % dir_path,
                            "--dataset-path", sequence_path,
                            "--cam-calib", os.path.join(dir_path, "../../data",
                                                        f"euroc_ds_calib_{args.setup.replace('-inertial', '')}.json"),
                            "--dataset-type", "euroc",
                            "--config-path", os.path.join(
                                dir_path, "../../data", f"euroc_config_{args.setup.replace('-', '_')}.json"),
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
                print(
                    f"Granite on sequence {sequence} run number {run_number + 1} FAILED")

            if failed:
                continue

            print(
                f"Calculating RMS ATE for {sequence} run number {run_number + 1}")

            # Calculate RMS ATE by using the evaluation script from TUM-RGBD (also used by ORB-SLAM3)
            evaluate_ate_scale_proc = subprocess.Popen(["python2", "-u",
                                                        os.path.join(
                                                            dir_path, "evaluate_ate_scale.py"),
                                                        new_gt_file_name,
                                                        result_file,
                                                        #"--max_difference", "0.1",
                                                        "--time_factor", "1e9",
                                                        "--plot", "plot.svg"],
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
                last_tracked_frame_t = float(
                    stdout.rstrip().split(',')[3])
                percents_tracked[run_number] = (
                    last_tracked_frame_t - first_frame_t) / (last_frame_t - first_frame_t) * 100.0

                print("RMS ATE: %f" % rmsate)
                print("percent_tracked: %f" % percents_tracked[run_number])

                if(rmsate > 1000 or percents_tracked[run_number] < 80.0):
                    failed = True
                    fail_count += 1
                    print(
                        f"Granite on sequence {sequence} run number {run_number + 1} FAILED")
                else:
                    rmsates[run_number] = rmsate

            except:
                print(stdout)

            if os.path.isfile(result_file):
                shutil.move(result_file, os.path.join(
                    sequence_output_path, f"{run_number + 1}_trajectory.txt"))
            if os.path.isfile("plot.svg"):
                shutil.move("plot.svg", os.path.join(
                    sequence_output_path, f"{run_number + 1}_plot.svg"))

        # get median of runs
        median = np.NaN
        if fail_count < args.runs_per_sequence:
            median = np.median(rmsates[rmsates != 0])

        # write statistics
        with open(summary_file, "a") as s_file:
            s_file.write(
                f"{sequence}: {median}, {fail_count}/{args.runs_per_sequence}, {np.max(percents_tracked)}\n")
        print("median RMS ATE of %s: %f" % (sequence, median))
        print("failed %d/%d" % (fail_count, args.runs_per_sequence))
