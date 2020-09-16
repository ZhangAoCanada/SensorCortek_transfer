import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle
import cv2
import math
import random
import pickle
import colorsys
import time

from sklearn import mixture
from sklearn.cluster import DBSCAN
from skimage.measure import find_contours
from collections import Counter

from glob import glob
from tqdm import tqdm

import multiprocessing as mp
from functools import partial

# detectron2 all classes
CLASS_NAMES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', \
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', \
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', \
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', \
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', \
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', \
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', \
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', \
        'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', \
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', \
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', \
        'hair drier', 'toothbrush']

ROAD_USERS = ['person', 'bicycle', 'car', 'motorcycle',
            'bus', 'train', 'truck', 'boat']

# Radar Configuration
RADAR_CONFIG_FREQ = 77 # GHz
DESIGNED_FREQ = 76.8 # GHz
RANGE_RESOLUTION = 0.1953125 # m
VELOCITY_RESOLUTION = 0.41968030701528203 # m/s
RANGE_SIZE = 256
DOPPLER_SIZE = 64
AZIMUTH_SIZE = 256
ANGULAR_RESOLUTION = np.pi / 2 / AZIMUTH_SIZE # radians
VELOCITY_MIN = - VELOCITY_RESOLUTION * DOPPLER_SIZE/2
VELOCITY_MAX = VELOCITY_RESOLUTION * DOPPLER_SIZE/2

def RandomColors(N, bright=True): 
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    # detectron uses random shuffle to give the differences
    random.seed(8888)
    random.shuffle(colors)
    return colors

def readRAD(radar_dir, frame_id):
    """
    Function:
        Get the 3D FFT results of the current frame.
    
    Args:
        radar_dir           ->          radar directory
        frame_id            ->          frame that is about to be processed.
    """
    if os.path.exists(os.path.join(radar_dir, "%.6d.npy"%(frame_id))):
        return np.load(os.path.join(radar_dir, "%.6d.npy"%(frame_id)))
    else:
        return None

def ReadRADMask(mask_dir, frame_i):
    """
    Function:
        Read the Range-Azimuth-Doppler mask from directory.

    Args:
        mask_dir            ->          directory that saves masks
    """
    filename = mask_dir + "RAD_mask_%.d.npy"%(frame_i)
    if os.path.exists(filename):
        RAD_mask = np.load(filename)
    else:
        RAD_mask = None
    return RAD_mask

def checkoutFormat(data_format, frame_id):
    """ Find out if the format is the right format """
    assert isinstance(data_format, list)
    assert all(isinstance(x, str) for x in data_format)
    assert (len(data_format) == 2 or len(data_format) == 3)
    if len(data_format) == 2:
        filename = ("%." + data_format[0] + data_format[1])%(frame_id)
    elif len(data_format) == 3:
        filename = (data_format[0] + "%." + data_format[1] + data_format[2])%(frame_id)
    return filename

def readRadarInstances(instance_dir, frame_id, data_format):
    """ read output radar instances. """
    filename = checkoutFormat(data_format, frame_id)
    pickle_file = os.path.join(instance_dir, filename)
    if os.path.exists(pickle_file):
        with open(pickle_file, "rb") as f:
            radar_instances = pickle.load(f)
        if len(radar_instances['classes']) == 0:
            radar_instances = None
    else:
        radar_instances = None
    return radar_instances
 
def getMagnitude(target_array, power_order=2):
    """
    Function:
        Get the magnitude of the complex array
    """
    target_array = np.abs(target_array)
    target_array = pow(target_array, power_order)
    return target_array 

def getLog(target_array, scalar=1., log_10=True):
    """
    Function:
        Get the log of the complex array
    """
    if log_10:
        return scalar * np.log10(target_array + 1.)
    else:
        return target_array

def WriteRadarObjPcl(radar_annotate_dir, radar_dict, frame_i):
    """
    Function:
        Write the annotated radar point cloud dictionary into the directory

    Args:
        radar_annotate_dir          ->          directory for saving
        frame_i                     ->          frame id
    """
    with open(radar_annotate_dir + "sensorcortek_" + str(frame_i) + ".pickle", "wb") as f:
        pickle.dump(radar_dict, f)

def readStereoImg(stereo_mrcnn_img_dir, frame_i):
    """
    Function:
        Read mask rcnn results on stereo left img
    
    Args:
        stereo_mrcnn_img_dir            ->          stereo mask rcnn result imgs
        frame_i                         ->          frame id
    """
    filename = os.path.join(stereo_mrcnn_img_dir, "stereo_input_%.d.jpg"%(frame_i))
    if os.path.exists(filename):
        img = cv2.imread(filename)[..., ::-1]
    else:
        img = None
    return img

def DbscanDenoise(pcl, epsilon=0.3, minimum_samples=100, n_jobs=1, dominant_op=False):
    """
    Function:
        Using DBSCAN for filtering out the noise data.

    Args:
        pcl             ->          point cloud to be denoised
        epsilon         ->          maximum distance to be considered as an object
        minimum_samples ->          miminum points to be considered as an object
    """
    clustering = DBSCAN(eps=epsilon, min_samples=minimum_samples, n_jobs=n_jobs).fit(pcl)
    output_labels = clustering.labels_
    if not dominant_op:
        output_pcl = []
        for label_i in np.unique(output_labels):
            if label_i == -1:
                continue
            output_pcl.append(pcl[output_labels == label_i])
    else:
        if len(np.unique(output_labels)) == 1 and np.unique(output_labels)[0] == -1:
            output_pcl = np.zeros([0,2])
            output_idx = []
        else:
            counts = Counter(output_labels)
            output_pcl = pcl[output_labels == counts.most_common(1)[0][0]]
            output_idx = np.where(output_labels == counts.most_common(1)[0][0])[0]
    return output_pcl

def AddonesToLstCol(target_array):
    """
    Function:
        Add ones to the last column of the target array

    Args:
        target_array            ->          array to be changed
    """
    adding_ones = np.ones([target_array.shape[0], 1])
    output_array = np.concatenate([target_array, adding_ones], axis=-1)
    return output_array

def CalibrateStereoToRadar(stereo_pcl, registration_matrix):
    """
    Function:
        Transfer stereo point cloud to radar frame.
    
    Args:
        stereo_pcl              ->          stereo point cloud
        registration_matrix     ->          registration matrix 
    """
    return np.matmul(AddonesToLstCol(stereo_pcl), registration_matrix)

def EuclideanDist(point1, point2):
    """
    Function:
        Calculate EuclideanDist
    """
    return np.sqrt(np.sum(np.square(point1 - point2), axis=-1))

def getSumDim(target_array, target_axis):
    """
    Function:
        Sum up one dimension of a  3D matrix.
    
    Args:
        column_index            ->          which column to be deleted
    """
    output = np.sum(target_array, axis=target_axis)
    return output 

def transfer2Scatter(RA_mask):
    """
    Function:
        Transfer RD indexes to pcl, for verifying quality

    Args:
        RA_mask              ->          Range-Azimuth mask
    """
    output_pcl = []
    for i in range(RA_mask.shape[0]):
        for j in range(RA_mask.shape[1]):
            if RA_mask[i, j] == 1:
                # point_range = ((RANGE_SIZE-1) - i) * RANGE_RESOLUTION
                # point_angle = (j - (AZIMUTH_SIZE/2)) * ANGULAR_RESOLUTION
                # ####################### Prince's method ######################
                point_range = ((RANGE_SIZE-1) - i) * RANGE_RESOLUTION
                point_angle = (j * (2*np.pi/AZIMUTH_SIZE) - np.pi) / \
                                (2*np.pi*0.5*RADAR_CONFIG_FREQ/DESIGNED_FREQ)
                point_angle = np.arcsin(point_angle)
                # ##############################################################
                point_zx = polarToCartesian(point_range, point_angle)
                output_pcl.append(np.array([[point_zx[1], point_zx[0]]]))
    if len(output_pcl) != 0:
        output_pcl = np.concatenate(output_pcl, axis=0)
    return output_pcl

def generateRACartesianImage(RA_img, RA_mask):
    """
    Function:
        Generate RA image in Cartesian Coordinate.

    Args:
        RA_img          ->          RA FFT magnitude
        RA_mask         ->          Mask generated 
    """
    output_img = np.zeros([RA_img.shape[0], RA_img.shape[0]*2])
    for i in range(RA_img.shape[0]):
        for j in range(RA_img.shape[1]):
            if RA_mask[i,j] == 1:
                point_range = ((RANGE_SIZE-1) - i) * RANGE_RESOLUTION
                # point_angle = (j - (AZIMUTH_SIZE/2)) * ANGULAR_RESOLUTION
                ####################### Prince's method ######################
                point_angle = (j * (2*math.pi/AZIMUTH_SIZE) - math.pi) / \
                                (2*math.pi*0.5*RADAR_CONFIG_FREQ/DESIGNED_FREQ)
                point_angle = math.asin(point_angle)
                ##############################################################
                point_zx = polarToCartesian(point_range, point_angle)
                new_i = int(output_img.shape[0] - \
                        np.round(point_zx[0]/RANGE_RESOLUTION)-1)
                new_j = int(np.round((point_zx[1]+50)/RANGE_RESOLUTION)-1)
                output_img[new_i,new_j] = RA_img[i,j] 
    norm_sig = plt.Normalize()
    # color mapping 
    output_img = plt.cm.viridis(norm_sig(output_img))
    output_img = output_img[..., :3]
    return output_img

def transferPcl2Cartesian(pcl):
    output_mask = np.zeros([RANGE_SIZE, RANGE_SIZE*2])
    for i in range(len(pcl)):
        pnt = pcl[i]
        point_zx = pnt[::-1]
        new_i = int(output_mask.shape[0] - \
                np.round(point_zx[0]/RANGE_RESOLUTION)-1)
        new_j = int(np.round((point_zx[1]+50)/RANGE_RESOLUTION)-1)
        output_mask[new_i,new_j] = 1.
    output_mask = np.where(output_mask > 0., 1., 0.)
    return output_mask

def transferMaskToCartesianMask(RA_mask):
    """
    Function:
        Transfer polar mask to cartesian mask.
    
    Args:
        RA_mask        ->          polar mask.
    """
    output_mask = np.zeros([RA_mask.shape[0], RA_mask.shape[0]*2])
    for i in range(RA_mask.shape[0]):
        for j in range(RA_mask.shape[1]):
            if RA_mask[i,j] == 1:
                point_range = ((RANGE_SIZE-1) - i) * RANGE_RESOLUTION
                # point_angle = (j - (AZIMUTH_SIZE/2)) * ANGULAR_RESOLUTION
                ####################### Prince's method ######################
                point_angle = (j * (2*np.pi/AZIMUTH_SIZE) - np.pi) / \
                                (2*np.pi*0.5*RADAR_CONFIG_FREQ/DESIGNED_FREQ)
                point_angle = np.arcsin(point_angle)
                ##############################################################
                point_zx = polarToCartesian(point_range, point_angle)
                new_i = int(output_mask.shape[0] - \
                        np.round(point_zx[0]/RANGE_RESOLUTION)-1)
                new_j = int(np.round((point_zx[1]+50)/RANGE_RESOLUTION)-1)
                output_mask[new_i,new_j] = 1.
    output_mask = np.where(output_mask > 0., 1., 0.)
    return output_mask

##################### coordinate transformation ######################
def cartesianToPolar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def polarToCartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

###################### PLOTTING FUNCTIONS START #########################
def prepareFigure(num_axes, figsize=None):
    assert num_axes <= 4
    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()
    if num_axes == 1:
        ax1 = fig.add_subplot(111)
        return fig, [ax1]
    if num_axes == 2: 
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        return fig, [ax1, ax2]
    if num_axes == 3:
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)
        return fig, [ax1, ax2, ax3]
    if num_axes == 4:
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        return fig, [ax1, ax2, ax3, ax4]

def clearAxes(ax_list):
    assert len(ax_list) >=1 
    plt.cla()
    for ax_i in ax_list:
        ax_i.clear()

def createColorList(class_names, class_list, all_colors):
    color_output_list = []
    for i in range(len(class_list)):
        current_class = class_list[i]
        color_i = np.array([all_colors[class_names.index(current_class)]])
        color_output_list.append(color_i)
    return color_output_list
 
def createLabelList(class_list, label_prefix):
    label_list = []
    for i in range(len(class_list)):
        current_class = class_list[i]
        label_list.append(label_prefix + "_" + current_class)
    return label_list

def pointScatter(bg_points, point_list, color_list, label_list, ax, xlimits, ylimits, title):
    assert len(point_list) == len(color_list)
    assert len(point_list) == len(label_list)
    if bg_points is not None:
        ax.scatter(bg_points[:, 0], bg_points[:, 1], s=0.2, c='blue', label="background")
    for i in range(len(point_list)):
        points_i = point_list[i]
        color_i = color_list[i]
        label_i = label_list[i]
        ax.scatter(points_i[:, 0], points_i[:,1], s=0.3, c=color_i, label=label_i)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.legend()
    ax.set_title(title)

def pclScatter(pcl_list, color_list, label_list, ax, xlimits, ylimits, title):
    assert len(pcl_list) == len(color_list)
    if label_list is not None:
        assert len(pcl_list) == len(label_list)
    for i in range(len(pcl_list)):
        pcl = pcl_list[i]
        color = color_list[i]
        if label_list == None:
            label = None
        else:
            label = label_list[i]
        ax.scatter(pcl[:, 0], pcl[:, 1], s=1, c=color, label=label)
    ax.set_xlim(xlimits)
    ax.set_ylim(ylimits)
    ax.set_title(title)

def imgPlot(img, ax, cmap, alpha, title=None):
    ax.imshow(img, cmap=cmap, alpha=alpha)
    if title == "RD":
        ax.set_xticks([0, 16, 32, 48, 63])
        ax.set_xticklabels([int(VELOCITY_MIN), int(VELOCITY_MIN)/2, 0, \
                            int(VELOCITY_MAX)/2, int(VELOCITY_MAX)])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("velocity (m/s)")
        ax.set_ylabel("range (m)")
    elif title == "RA":
        ax.set_xticks([0, 64, 128, 192, 255])
        ax.set_xticklabels([-90, -45, 0, 45, 90])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("angle (degrees)")
        ax.set_ylabel("range (m)")
    elif title == "RAD mask in cartesian":
        ax.set_xticks([0, 128, 256, 384, 512])
        ax.set_xticklabels([-50, -25, 0, 25, 50])
        ax.set_yticks([0, 64, 128, 192, 255])
        ax.set_yticklabels([50, 37.5, 25, 12.5, 0])
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
    else:
        ax.axis('off')
    if title is not None:
        ax.set_title(title)

def keepDrawing(fig, time_duration):
    fig.canvas.draw()
    plt.pause(time_duration)

def norm2Image(array):
    norm_sig = plt.Normalize()
    img = plt.cm.viridis(norm_sig(array))
    img *= 255.
    img = img.astype(np.uint8)
    return img

def applyMask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                 image[:, :, c] *
                                (1 - alpha) + alpha * color[c] * 255,
                                 image[:, :, c])
    return image

def drawContour(mask, axe, color):
    """
    Draw mask contour onto the image.
    """
    mask_padded = np.zeros((mask.shape[0]+2, mask.shape[1]+2), dtype=np.uint8)
    mask_padded[1:-1, 1:-1] = mask
    contours = find_contours(mask_padded, 0.1, fully_connected='low')
    for verts in contours:
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        axe.add_patch(p)

def mask2BoxOrEllipse(mask, mode="box", n_jobs=1):
    """
    Find bounding box from mask
    """
    idxes = []
    output = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 1:
                idxes.append([i, j])
    idxes = np.array(idxes)
    idx_clusters = DbscanDenoise(idxes, epsilon=20, minimum_samples=3, n_jobs=n_jobs)
    if mode == "box":
        for cluster_id in range(len(idx_clusters)):
            current_cluster = idx_clusters[cluster_id]
            x_min = np.amin(current_cluster[:, 0])
            x_max = np.amax(current_cluster[:, 0])
            y_min = np.amin(current_cluster[:, 1])
            y_max = np.amax(current_cluster[:, 1])
            output.append([x_min, x_max, y_min, y_max])
        if len(output) == 0:
            output = None
        else:
            output = np.array(output)
            x_min = np.amin(output[:, 0])
            x_max = np.amax(output[:, 1])
            for i in range(len(output)):
                output[i, 0] = x_min
                output[i, 1] = x_max
        return output
    elif mode == "ellipse":
        for cluster_id in range(len(idx_clusters)):
            current_cluster = idx_clusters[cluster_id]
            cluster_mean, cluster_cov = GaussianModel(current_cluster)
            output.append([cluster_mean, cluster_cov])
        if len(output) == 0:
            output = None
        return output
    else:
        raise ValueError("Wrong input parameter ------ mode")

def drawBoxOrEllipse(inputs, class_name, axe, color, mode="box"):
    """
    Draw bounding box onto the image.
    """
    if mode == "box":
        for box in inputs:
            y1, y2, x1, x2 = box
            r = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1.5,
                        alpha=0.5, linestyle="dashed", edgecolor=color,
                        facecolor="none")
            axe.add_patch(r)

        # axe.text(x1+2, y1-3, class_name, size=5, verticalalignment='baseline',
        axe.text(x1+1, y1-3, class_name, size=10, verticalalignment='baseline',
                color='w', backgroundcolor="none",
                bbox={'facecolor': color, 'alpha': 0.5,
                    'pad': 2, 'edgecolor': 'none'})
    elif mode == "ellipse":
        for e in inputs:
            mean, cov = e[0], e[1]
            mean = np.flip(mean)
            cov = np.flip(cov)
            x1, y1 = mean
            ell = getEllipse(color, mean, cov, scale_factor=5)
            axe.add_patch(ell)
        axe.text(x1, y1, class_name, size=5, verticalalignment='center',
                color='w', backgroundcolor="none",
                bbox={'facecolor': color, 'alpha': 0.5,
                        'pad': 2, 'edgecolor': 'none'})
    else:
        raise ValueError("Wrong input parameter ------ mode")

def getEllipse(color, means, covariances, scale_factor=1):
    """
    Function:
        Draw 2D Gaussian Ellipse.

    Args:
        means           ->          center of the Gaussian
        covariances     ->          covariance of Gaussian
    """
    sign = np.sign(means[0] / means[1])
    eigen, eigen_vec = np.linalg.eig(covariances)

    eigen_root_x = np.sqrt(eigen[0]) * scale_factor
    eigen_root_y = np.sqrt(eigen[1]) * scale_factor
    theta = np.degrees(np.arctan2(*eigen_vec[:,0][::-1]))

    ell = Ellipse(xy = (means[0], means[1]), width = eigen_root_x,
                height = eigen_root_y, angle = theta, \
                facecolor = 'none', edgecolor = color)
    return ell

def saveFigure(save_dir, name_prefix, frame_i):
    plt.savefig(save_dir + name_prefix + str(frame_i) + ".png")
###################### PLOTTING FUNCTIONS END #########################

def main_formp(frame_i, stereo_img_dir, radar_ral_dir, radar_annotate_dir, mask_dir, \
                frame_delay, distance_threshold, name_prefix, fig, axes, \
                all_colors, if_plot, if_save, n_jobs=1):
    RAD_FFT = readRAD(os.path.join(radar_ral_dir, "RAD_numpy"), frame_i)
    left_img = readStereoImg(stereo_img_dir, frame_i+frame_delay)
    radar_dict = readRadarInstances(radar_annotate_dir, frame_i, ["radar_obj_", "d", ".pickle"])
    radar_RAD_mask = ReadRADMask(mask_dir, frame_i)
    if RAD_FFT is None or left_img is None or radar_dict is None or \
                                    radar_RAD_mask is None:
        print("Frame %d has no object detected."%(frame_i))
        if RAD_FFT is None:
            print("RAD data not loaded.")
        if left_img is None:
            print("stereo left image not loaded.")
        if radar_dict is None:
            print("radar annotation not loaded.")
        if radar_RAD_mask is None:
            print("radar RAD mask not loaded.")
        pass
    else:
        RAD_FFT = getMagnitude(RAD_FFT,  power_order=2)
        RD_img = norm2Image(getLog(getSumDim(RAD_FFT, 1), scalar=10, log_10=True))
        RA_img = norm2Image(getLog(getSumDim(RAD_FFT, -1), scalar=10, log_10=True))

        radar_classes = radar_dict["classes"] 
        radar_masks = radar_dict["masks"]
        radar_box3D = radar_dict["boxes"]
        # stereo_correspond_masks = radar_dict["stereo_masks"]

        if if_save:
            WriteRadarObjPcl(radar_annotate_dir, radar_dict, frame_i)

        if if_plot:
            clearAxes(axes)

            ellipses = []
            RA_cart_img = np.zeros([RAD_FFT.shape[0], \
                                RAD_FFT.shape[0]*2])
            RA_cart_img = norm2Image(RA_cart_img)
            RA_all_mag = getLog(getSumDim(RAD_FFT, -1))
            RA_original_mask = np.where(getSumDim(radar_RAD_mask, -1) > 0, 1., 0.)

            for class_id in range(len(radar_classes)):
                obj_class = radar_classes[class_id]
                # if obj_class in ["person", "bicycle", "motorcycle"]:
                    # obj_class = "person"
                # elif obj_class in ["car", "bus", "truck"]:
                    # obj_class = "car"
                class_num = CLASS_NAMES.index(obj_class)
                class_color = all_colors[class_num]
                obj_masks = radar_masks[class_id]
                ######## draw stereo mask and box #######
                # stereo_cor_mask = stereo_correspond_masks[class_id]
                # applyMask(left_img, stereo_cor_mask, class_color)
                # stereo_box = mask2BoxOrEllipse(stereo_cor_mask, "box", \
                                                # n_jobs=n_jobs)
                # drawBoxOrEllipse(stereo_box, obj_class, axes[0], class_color, "box")
                ########################################
                obj_masks_combined = obj_masks
                obj_masks_combined = np.where(obj_masks_combined > 0, 1., 0.)
                RD_mask = np.where(getSumDim(obj_masks_combined, 1) >= 1, 1, 0)
                RA_mask = np.where(getSumDim(obj_masks_combined, -1) >= 1, 1, 0)
                RA_original_mask -= RA_mask
                RA_cart_mask = transferMaskToCartesianMask(RA_mask)
                applyMask(RD_img, RD_mask, class_color)
                applyMask(RA_img, RA_mask, class_color)
                applyMask(RA_cart_img, RA_cart_mask, class_color)
                RA_cartesian = transfer2Scatter(RA_mask)
                if len(RA_cartesian) == 0:
                    continue
                ############# draw mask contour ################
                drawContour(RA_cart_mask, axes[1], class_color)
                drawContour(RD_mask, axes[2], class_color)
                drawContour(RA_mask, axes[3], class_color)
                ############# draw box or ellipses ################
                mode = "box" # either "box" or "ellipse"
                RD_box = mask2BoxOrEllipse(RD_mask, mode, n_jobs=n_jobs)
                RA_box = mask2BoxOrEllipse(RA_mask, mode, n_jobs=n_jobs)
                RA_cart_box = mask2BoxOrEllipse(RA_cart_mask, mode, \
                                                n_jobs=n_jobs)
                if RD_box is None or RA_box is None or RA_cart_box is None:
                    continue
                drawBoxOrEllipse(RA_cart_box, obj_class, axes[1], class_color, mode)
                drawBoxOrEllipse(RD_box, obj_class, axes[2], class_color, mode)
                drawBoxOrEllipse(RA_box, obj_class, axes[3], class_color, mode)

            RA_original_mask = np.where(RA_original_mask > 0, 1., 0.)
            RA_rest = transferMaskToCartesianMask(RA_original_mask)
            applyMask(RA_cart_img, RA_rest, [1,1,1])
            
            imgPlot(left_img, axes[0], None, 1,  "mrcnn")
            imgPlot(RA_cart_img, axes[1], None, 1, "RAD mask in cartesian")
            imgPlot(RD_img, axes[2], None, 1, "RD")
            imgPlot(RA_img, axes[3], None, 1, "RA")
            # keepDrawing(fig, 0.1)
            saveFigure(radar_annotate_dir, name_prefix, frame_i)

def main_mp(stereo_img_dir, radar_ral_dir, radar_annotate_dir, radar_RAD_mask_dir, \
                                                        sequence, if_plot, if_save):
    ##### frame delay between two sensors (not sure whether it exists) #####
    frame_delay = 0
    distance_threshold = 5.0
    # name_prefix = "test_"
    name_prefix = "sensorcortek_"

    if if_plot:
        # fig, axes = prepareFigure(4, figsize=(25,15))
        fig, axes = prepareFigure(4, figsize=(20,10))
    else:
        fig, axes = prepareFigure(2)

    all_colors = RandomColors(len(CLASS_NAMES))

    pool = mp.Pool(6)

    main_formp_1arg = partial(main_formp, 
                                stereo_img_dir = stereo_img_dir,
                                radar_ral_dir = radar_ral_dir,
                                radar_annotate_dir = radar_annotate_dir,
                                mask_dir = radar_RAD_mask_dir,
                                frame_delay = frame_delay,
                                distance_threshold = distance_threshold,
                                name_prefix = name_prefix,
                                fig = fig,
                                axes = axes,
                                all_colors = all_colors,
                                if_plot = if_plot,
                                if_save = if_save)
    
    for _ in tqdm(pool.imap_unordered(main_formp_1arg, \
                    range(sequence[0], sequence[1])), \
                    total = sequence[1] - sequence[0]):
        pass
    pool.close()
    pool.join()
    pool.close()

if __name__ == "__main__":
    time_stamp = "2020-09-03-12-30-11"
    stereo_img_dir = "/home/ao/Documents/stereo_radar_calibration_annotation/annotation/stereo_input"
    radar_ral_dir = "/DATA/" + time_stamp + "/ral_outputs_" + time_stamp + "/"
    radar_annotate_dir = "/home/ao/Documents/stereo_radar_calibration_annotation/annotation/radar_annotation/"
    radar_RAD_mask_dir = "/home/ao/Documents/stereo_radar_calibration_annotation/radar_process/radar_ral_process/radar_RAD_mask/"
    sequence = [0, 3000]
    if_plot = True
    save_gt = False
    main_mp(stereo_img_dir, radar_ral_dir, radar_annotate_dir, radar_RAD_mask_dir, \
                        sequence, if_plot=if_plot, if_save=save_gt)
