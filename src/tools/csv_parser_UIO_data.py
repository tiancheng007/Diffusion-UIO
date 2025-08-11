#!/usr/bin/env python3

import numpy as np
import os
import glob
from tqdm import tqdm
import csv

def process_csv_file(csv_path, horizon):
    with open(csv_path) as f:
        csv_reader = csv.reader(f, delimiter=",")
        odometry = []
        vy_gts = []
        yaw_rate_gts = []
        column_idxs = dict()
        started = False
        for row in csv_reader:
            if len(column_idxs) == 0:
                for i in range(len(row)):
                    column_idxs[row[i].split("(")[0]] = i
                continue
            vx = float(row[column_idxs["vx_ms"]])
            vy = float(row[column_idxs["vy_ms"]])
            ax = float(row[column_idxs["ax_ms2"]])
            ay = float(row[column_idxs["ay_ms2"]])
            Tw = float(row[column_idxs["tw_nm"]])
            Fy1_db = float(row[column_idxs["Fy1_db"]])
            Fy2_db= float(row[column_idxs["Fy2_db"]])
            vy_db = float(row[column_idxs["vy_db"]])
            vtheta = float(row[column_idxs["yaw_rads"]])
            steering = float(row[column_idxs["st_ang_rad"]])
            miu = float(row[column_idxs["miu"]])

            # Please note that the variables ay, Fy1_db, Fy2_db, vy_db, ax, miu, and Tw are not required for Diff-UIO.
            # You may modify the dataset generation code as needed; however, you must also update dataset_UIO.py
            # to ensure that the dataset input format is aligned accordingly.
            odometry.append(np.array([vx, vtheta, ay, Fy1_db, Fy2_db, vy_db, ax, miu, Tw, steering]))

            if started:
                vy_gts.append(vy)
                yaw_rate_gts.append(vtheta)
            started = True
        odometry = np.array(odometry)
        vy_gts = np.array(vy_gts)
        yaw_rate_gts = np.array(yaw_rate_gts)

        return odometry, vy_gts, yaw_rate_gts


def compile_dataset(data_dir, horizon):
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

    all_features = []
    all_labels = []

    for csv_file in csv_files:
        print(f"Processing {csv_file}")
        odometry, vy_gts, yaw_rate_gts = process_csv_file(csv_file, horizon)

        features = np.zeros((len(vy_gts)-1, 10), dtype=np.double)
        labels = np.zeros((len(vy_gts)-1, 4), dtype=np.double)
        for i in tqdm(range(len(vy_gts)-1), desc=f"Compiling dataset from {csv_file}"):
            features[i] = np.array([odometry[i].T])
            labels[i] = np.array([vy_gts[i], yaw_rate_gts[i], vy_gts[i+1], yaw_rate_gts[i+1]])


        all_features.append(features)
        all_labels.append(labels)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("Final features shape:", all_features.shape)
    print("Final labels shape:", all_labels.shape)

    np.savez(os.path.join(data_dir, "data_smooth_004_55_115_test_h1.npz"), features=all_features, labels=all_labels)


if __name__ == "__main__":
    data_dir = './data'
    horizon = 1
    compile_dataset(data_dir, horizon)