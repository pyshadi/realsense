
'''

'''

import sys
import math
import time
import logging
import json
import open3d as o3d
import numpy as np
import cv2
import pyrealsense2 as rs

from shapely.geometry import Polygon, JOIN_STYLE
from scipy import spatial
from descartes import PolygonPatch


# image parameters
width = 424
height = 240

DS5_product_ids = ["0AD1", "0AD2", "0AD3", "0AD4", "0AD5", "0AF6",
                   "0AFE", "0AFF", "0B00", "0B01", "0B03", "0B07", "0B3A"]


def save_dict_to_json(fname, my_dict):
    with open(fname, 'w') as fp:
        json.dump(my_dict, fp, indent=4)

def find_device_that_supports_advanced_mode(ctx, devices):
    for dev in devices:
        if dev.supports(rs.camera_info.product_id) and str(dev.get_info(rs.camera_info.product_id)) in DS5_product_ids:
            if dev.supports(rs.camera_info.name):
                logging.info("Found device that supports advanced mode: %r", dev.get_info(rs.camera_info.name))
            return dev
    return None

def enable_advanced_mode(advnc_mode):
    """Attempts to enable advanced mode
    """
    # Loop until we successfully enable advanced mode
    while not advnc_mode.is_enabled():
        logging.info("Trying to enable advanced mode...")
        advnc_mode.toggle_advanced_mode(True)
        # At this point the device will disconnect and re-connect.
        logging.info("Device disconnecting. Sleeping for 5 seconds...")
        time.sleep(5)
        # The 'dev' object will become invalid and we need to initialize it again
        dev = find_device_that_supports_advanced_mode()
        if dev is None:
            logging.error("Device did not reconnect! Exiting")
            sys.exit(1)
        advnc_mode = rs.rs400_advanced_mode(dev)
        logging.info("Advanced mode is %r", "enabled" if advnc_mode.is_enabled() else "disabled")

    return advnc_mode

def load_setting_file(ctx, devices, setting_file):
    """Loads a setting file

    Arguments:
        ctx {ctx} -- RS context
        devices {device} -- Realsense device
        setting_file {str} -- Path to settings file

    Returns:
        bool -- True if successful
    """
    dev = find_device_that_supports_advanced_mode(ctx, devices)
    if dev is None:
        logging.error("No device supports the advanced mode! Can not upload settings file: %r", setting_file)
        return None
    advnc_mode = rs.rs400_advanced_mode(dev)
    logging.info("Advanced mode is %r", "enabled" if advnc_mode.is_enabled() else "disabled")
    advnc_mode = enable_advanced_mode(advnc_mode)
    # Read settings file as a string
    with open(setting_file, 'r') as file:
        settings_json_str = file.read()
    advnc_mode.load_json(settings_json_str)
    return True

def get_intrinsics(pipeline, stream=rs.stream.color):
    """Get intrinics for specified stream

    Arguments:
        pipeline {rs::pipeline} -- The pipeline that has been configured

    Keyword Arguments:
        stream {rs::stream::type} -- Stream Type (default: {rs.stream.color})

    Returns:
        rs::intrinsics -- The instrinsics object
    """
    streams = [stream_ for stream_ in pipeline.get_active_profile().get_streams() if stream_.stream_type() == stream]
    intrinsics = None
    if streams:
        intrinsics = streams[0].as_video_stream_profile().get_intrinsics()
    return intrinsics

def create_projection_matrix(intrinsics):

    fx, fy, ppx, ppy = intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy
    proj_mat = np.array([[fx, 0, ppx, 0], [0, fy, ppy, 0], [0, 0, 1, 0]])
    return proj_mat

def project_points_img(points, proj_mat, width, height):
    """Projects points into image given a projection matrix

    Arguments:
        points {ndarray} -- 3D points
        proj_mat {ndarray, 3X4} -- Projection Matrix
        width {int} -- width of image
        height {height} -- height of image

    Returns:
        ndarray -- pixels
    """
    pixels = proj_mat.dot(points)
    pixels = np.divide(pixels[:2, :], pixels[2, :]).transpose().astype(np.int)

    # Remove pixels that are outside the image
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height)
    # mask_x = (pixels[:, 0] < width) & (pixels[:, 0] > 0)
    # mask_y = (pixels[:, 1] < height) & (pixels[:, 1] > 0)

    # # Return the pixels and points that are inside the image
    # pixels = pixels[mask_x & mask_y]
    return pixels

def get_pix_coordinates(pts, proj_mat, w, h):
    """Get Pixel coordinates of ndarray

    Arguments:
        pts {ndarray} -- 3D point clouds 3XN
        proj_mat {ndarray} -- 4X3 Projection Matrix
        w {int} -- width
        h {int} -- height

    Returns:
        ndarray -- Pixel coordinates
    """
    points_t = np.ones(shape=(4, pts.shape[1]))
    points_t[:3, :] = pts
    pixels = project_points_img(points_t, proj_mat, w, h)

    return pixels
