#!/usr/bin/env python3

import argparse
import subprocess
import csv
import os.path
from send2trash import send2trash
import numpy as np
import shutil

sequences = [
    # "A-0",
    # "A-1",
    # "A-2",
    # "A-3",
    # "A-4",
    # "A-5",
    # "A-6",
    # "B-0",
    # "B-1",
    # "B-2",
    # "B-3",
    # "B-4",
    # "B-5",
    # "B-6",
    # "B-7",
    # "C-0",
    # "C-1",
    # "C-2",
    "D-0",
    # "D-1",
    # "D-2",
    # "D-3",
    "D-4",
    # "E-0",
    # "E-1",
    # "E-2",
    # "F-0",
    # "F-1",
    # "F-2",
    # "F-3",
    # "F-4",
    # "F-5",
    # "G-0",
    # "G-1",
    # "G-2",
    # "H-0"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    This script runs over many sequences (specified in 'sequences') of the Morocco dataset. Every sequence is executed 'runs_per_sequence' times. The evaluation script from TUM-RGBD (also used by ORB-SLAM3) is used to calculate the Root Mean Square Absolute Trajectory Error (RMS ATE). The median of all runs is reported in 'rmsate_summary.txt'.
    ''')
    parser.add_argument("--dataset_path", help="Path to the Morocco dataset.")
    parser.add_argument(
        "--output_path", help="Path to to where the evaluation results should be stored.")
    parser.add_argument('--runs_per_sequence', type=int,
                        help='How often should every sequence be evaluated. Default: 3', default=3)
    parser.add_argument('--setup',
                        help='Either "mono", "stereo-inertial" or "stereo"', default="stereo-inertial")
    args = parser.parse_args()

    config_path = os.path.join(args.dataset_path, "config", "Granite")

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

    # run over all sequences
    for sequence in sequences:

        sequence_output_path = os.path.join(
            args.output_path, f"{sequence}_{args.setup}")
        if not os.path.isdir(sequence_output_path):
            os.mkdir(sequence_output_path)

        sequence_path = os.path.join(args.dataset_path, sequence)

        print("Looking for a sequence in %s" % sequence_path)

        # initialize statistics
        rmsates = np.zeros(args.runs_per_sequence, dtype=np.float64)
        percents_tracked = np.zeros(args.runs_per_sequence, dtype=np.float64)
        fail_count = 0

        with open(os.path.join(sequence_path, "timestamps.txt")) as timestamps_file:
            lines = timestamps_file.readlines()
            first_frame_t = int(lines[0])
            last_frame_t = int(lines[-1])

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
                            "--cam-calib", os.path.join(config_path,
                                                        f"calibration_{args.setup.replace('-inertial', '')}_high-res.json"),
                            "--dataset-type", "euroc",
                            "--config-path", os.path.join(
                                config_path, f"config_{args.setup.replace('-', '_')}_new.json"),
                            #"--marg-data", "morocco_marg_data",
                            "--show-gui", "0",
                            "--result-path", "result.json",
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
                                                        os.path.join(
                                                            sequence_path, "mav0", "gt", "data.csv"),
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

            shutil.move(result_file, os.path.join(
                sequence_output_path, f"{run_number + 1}_trajectory.txt"))
            if os.path.isfile(os.path.join(dir_path, "result.json")):
                shutil.move(os.path.join(dir_path, "result.json"), os.path.join(
                    sequence_output_path, f"{run_number + 1}_result.json"))
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
