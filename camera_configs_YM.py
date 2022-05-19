# Forest

import cv2
import numpy as np

left_camera_matrix = np.array([[413.4549, -0.4288, 347.1880],
                               [0, 413.1756, 244.0289],
                               [0, 0, 1]])
left_distortion = np.array([[-0.0595, 0.2561, 0.0020, 0.0041, -0.2901]])

right_camera_matrix = np.array([[415.5471, -0.2865, 344.3784],
                                [0, 414.5074, 239.6630],
                                [0, 0, 1]])

right_distortion = np.array([[-0.0478, 0.1811, 0.0016, 0.0069, -0.1705]])

R = np.matrix([
    [1.0000, -0.0010, 0.0041],
    [0.0010, 1.0000, -0.0006],
    [-0.0041, 0.0006, 1.0000],
])

# print(R)

T = np.array([-120.5154, 0.1670, -1.6946])  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
