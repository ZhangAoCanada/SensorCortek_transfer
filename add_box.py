import os
import cv2
import numpy as np
import math
import pickle
import random
import colorsys
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from collections import Counter

from glob import glob
from tqdm import tqdm



def readRadarInstances(instance_dir, frame_i):
    filename = os.path.join(instance_dir, "radar_obj_%.d.pickle"%(frame_i))
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            radar_instances = pickle.load(f)
        if len(radar_instances['classes']) == 0:
            radar_instances = None
    else:
        radar_instances = None
    return radar_instances
 
def RAD2bbox3D(radar_masks):
    """ Transfer RAD masks to 3D bounding boxes """
    bbox = []
    for mask_i in range(len(radar_masks)):
        mask = radar_masks[mask_i]
        indexes = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                for k in range(mask.shape[2]):
                    if mask[i,j,k] > 0:
                        indexes.append([i, j, k])
        indexes = np.array(indexes)
        x_min, x_max = np.amin(indexes[:, 0]), np.amax(indexes[:, 0])+1
        y_min, y_max = np.amin(indexes[:, 1]), np.amax(indexes[:, 1])+1
        z_min, z_max = np.amin(indexes[:, 2]), np.amax(indexes[:, 2])+1
        bbox.append([x_min, x_max, y_min, y_max, z_min, z_max])
    return np.array(bbox)


def writeRadarObjPcl(instance_dir, radar_dict, frame_i):
    filename = os.path.join(instance_dir, "radar_obj_%.d.pickle"%(frame_i))
    with open(filename, "wb") as f:
        pickle.dump(radar_dict, f)


def main():
    annot_dir = "./stereo_radar_calibration_annotation/annotation/radar_annotation"
    for i in tqdm(range(0, 3000)):
        radar_instances = readRadarInstances(annot_dir, i)
        if radar_instances is None:
            continue

        radar_masks = radar_instances["masks"] 
        radar_classes = radar_instances["classes"] 
        radar_boxes = radar_instances["boxes"]

        radar_dict = {}
        radar_dict["classes"] = radar_classes
        radar_dict["masks"] = radar_masks
        radar_dict["boxes"] = RAD2bbox3D(radar_masks)

        writeRadarObjPcl(annot_dir, radar_dict, i)

if __name__ == "__main__":
    main()
