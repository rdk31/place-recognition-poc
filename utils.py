import os
import glob
import numpy as np


def parse_alog_file(alog_path):
    gps_and_image_info = []
    coord_n = None
    coord_e = None
    with open(alog_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if "GPS" in line:
                parts = line.strip().split()[-1].split(",")
                parts_dict = {part.split("=")[0]: part.split("=")[1] for part in parts}
                coord_n = float(parts_dict.get("N", 0))
                coord_e = float(parts_dict.get("E", 0))
            if "LADYBUG_GRAB" in line:
                parts = line.strip().split()[-1].split(",")
                image_files = [
                    part.split("=")[1].split("/")[-1]
                    for part in parts
                    if "File" in part
                ]
                if (
                    coord_n is not None and coord_e is not None
                ):  # only append if we have valid GPS coordinates
                    gps_and_image_info.append(((coord_n, coord_e), image_files))
                coord_n = None
                coord_e = None

    return gps_and_image_info


def get_data(root_dir):
    raw_logs_dir = os.path.join(root_dir, "Raw_Logs")

    train_alog_files = glob.glob(
        os.path.join(raw_logs_dir, "CarData_24_7_2008_____13_54_03/*.alog")
    )

    train_data = []
    for alog_file in train_alog_files:
        train_data.extend(parse_alog_file(alog_file))

    test_alog_files = glob.glob(
        os.path.join(raw_logs_dir, "CarData_24_7_2008_____15_10_45/*.alog")
    )
    test_data = []
    for alog_file in test_alog_files:
        test_data.extend(parse_alog_file(alog_file))

    return train_data, test_data


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2)) ** 2))
