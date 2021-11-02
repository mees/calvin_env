import math

import numpy as np


class Camera:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def distance_map_to_point_cloud(self, distances, fov, width, height):
        """Converts from a depth map to a point cloud.
        Args:
          distances: An numpy array which has the shape of (height, width) that
            denotes a distance map. The unit is meter.
          fov: The field of view of the camera in the vertical direction. The unit
            is radian.
          width: The width of the image resolution of the camera.
          height: The height of the image resolution of the camera.
        Returns:
          point_cloud: The converted point cloud from the distance map. It is a numpy
            array of shape (height, width, 3).
        """
        f = height / (2 * math.tan(fov / 2.0))
        px = np.tile(np.arange(width), [height, 1])
        x = (2 * (px + 0.5) - width) / f * distances / 2
        py = np.tile(np.arange(height), [width, 1]).T
        y = (2 * (py + 0.5) - height) / f * distances / 2
        point_cloud = np.stack((x, y, distances), axis=-1)
        return point_cloud

    def z_buffer_to_real_distance(self, z_buffer, far, near):
        """Function to transform depth buffer values to distances in camera space"""
        return 1.0 * far * near / (far - (far - near) * z_buffer)

    def process_rgbd(self, obs, nearval, farval):
        (width, height, rgbPixels, depthPixels, segmentationMaskBuffer) = obs
        rgb = np.reshape(rgbPixels, (height, width, 4))
        rgb_img = rgb[:, :, :3]
        depth_buffer = np.reshape(depthPixels, [height, width])
        depth = self.z_buffer_to_real_distance(z_buffer=depth_buffer, far=farval, near=nearval)
        return rgb_img, depth
