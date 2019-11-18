import cv2
import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow, figure
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import display, Math, Markdown
paths_nums = range(1, 5)  # 21
paths = ["/home/mypc/bgu/vision/computational_vision/assignment1/photos/{}.jpeg".format(num) for num in paths_nums]
world_points_list = []
corners_list = []
shape=None
images = []
for path in paths:
    # print path
    chess_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    images.append(chess_img)
    # cv2.imshow("bla", chess_img)
    # cv2.waitKey(0)

    pattern_size = (7, 7)
    found, corners = cv2.findChessboardCorners(chess_img, pattern_size)
    corners_list.append(corners)
    # print(corners)
    # print(found)
    color = cv2.cvtColor(chess_img, cv2.COLOR_GRAY2RGB)
    cv2.drawChessboardCorners(color, pattern_size, corners, found)
    # cv2.imshow('img', color)
    # cv2.waitKey(0)
    # imshow(with_corners, figure=figure(figsize=(10, 10)))

    from itertools import product

    xs, ys = pattern_size
    world_points = np.array(
        [(x, y, 0) for y, x in product(range(ys), range(xs))],
                            dtype=np.float32)

    world_points_list.append(world_points)
    shape = chess_img.shape
    # print(world_points)

retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
    objectPoints=world_points_list,
    imagePoints=corners_list,
    imageSize=shape[::-1],
    cameraMatrix=None,
    distCoeffs=None)

np.set_printoptions(suppress=True)
# print(cameraMatrix)

print 'tvecs'
print tvecs[0]
print rvecs[0]


########################################### D ################################################

r_matrix, _ = cv2.Rodrigues(np.array(rvecs[0]))
row, col = 3, 3
f_mat = np.zeros((4, 4))
f_mat[:row, :col] = r_matrix
f_mat[:row, col] = np.array(tvecs[0]).flatten()
f_mat[row, col] = 1


k_mat = np.zeros((4, 4))
k_mat[:row, :col] = cameraMatrix
f_mat[row, col] = 1

# print 'f_mat'
# print f_mat
#
# print "rod"
# print r_matrix


m_mat = k_mat * f_mat
print 'm_mat'
print m_mat

print
########################################### E ################################################
# a = np.zeros((2, 49))
# print len(world_points_list[0])
image_num = 0
projected_points, _ = cv2.projectPoints(world_points_list[image_num], rvecs[image_num], tvecs[image_num], cameraMatrix, distCoeffs)
# print 'a'
# print a
#
# print 'corners_list[0]'
# print corners_list[0]
errors = [((x-y)/x).tolist() for x, y in zip(corners_list[0], projected_points)]
# print x


########################################### F ################################################
image_num = 1
cube_world_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
], dtype=float)
print 'world_points_list[image_num]'
print world_points_list[image_num]
print
print
print
print
print 'cube_world_points'
print cube_world_points
cube_projected_point, _ = cv2.projectPoints(cube_world_points, rvecs[image_num], tvecs[image_num], cameraMatrix, distCoeffs)
print
print
print
print
print
print 'projected_points_1'
print cube_projected_point

# rotating
teta = np.pi/3
rotation_mat = np.array([
    [np.cos(teta), -np.sin(teta), 0],
    [np.sin(teta), np.cos(teta), 0],
    [0, 0, 1]
], dtype=float)


def to_homogonos(vec):
    arr = np.ones(len(vec) + 1, dtype=float)
    arr[0:len(vec)] = vec
    return arr

homogonos_projected_points_1 = np.array([to_homogonos(x[0]) for x in cube_projected_point]).transpose()
# homogonos_projected_points_1 = np.array([to_homogonos(x[0]) for x in projected_points_1])
traslatation_mat = np.array([
    [1, 0, -cube_projected_point[0][0][0]],
    [0, 1, -cube_projected_point[0][0][1]],
    [0, 0, 1]
], dtype=float)

traslatation_back_mat = np.array([
    [1, 0, cube_projected_point[0][0][0]],
    [0, 1, cube_projected_point[0][0][1]],
    [0, 0, 1]
], dtype=float)

# rotated_points = [rotation_mat * x[0] for x in projected_points_1]
rotated_points = [x.dot(rotation_mat) for x in homogonos_projected_points_1]
# rotated_points_hom = (traslatation_back_mat * rotation_mat * traslatation_mat).dot(homogonos_projected_points_1)
# rotated_points_hom = rotated_points_hom.transpose()
# rotated_points = [np.array(x[0:2]) for x in rotated_points_hom]
image_1 = images[image_num]
# print rotated_points[0]
# to_tuple = lambda arr: tuple([int(x) for x in arr[0]])  # this version is for the drawing without rotation
to_tuple = lambda arr: tuple([int(x) for x in arr])
# cv2.line(image_1, (1, 3), (2, 4), color=3)
cv2.line(image_1, to_tuple(rotated_points[0]), to_tuple(rotated_points[0]), color=3)
# drawing lines
cv2.line(image_1, to_tuple(rotated_points[0]), to_tuple(rotated_points[2]), color=3)
cv2.line(image_1, to_tuple(rotated_points[0]), to_tuple(rotated_points[4]), color=3)

cv2.line(image_1, to_tuple(rotated_points[1]), to_tuple(rotated_points[3]), color=3)
cv2.line(image_1, to_tuple(rotated_points[1]), to_tuple(rotated_points[5]), color=3)

cv2.line(image_1, to_tuple(rotated_points[5]), to_tuple(rotated_points[4]), color=3)
cv2.line(image_1, to_tuple(rotated_points[5]), to_tuple(rotated_points[7]), color=3)

cv2.line(image_1, to_tuple(rotated_points[7]), to_tuple(rotated_points[6]), color=3)
cv2.line(image_1, to_tuple(rotated_points[7]), to_tuple(rotated_points[3]), color=3)

cv2.line(image_1, to_tuple(rotated_points[4]), to_tuple(rotated_points[6]), color=3)
cv2.line(image_1, to_tuple(rotated_points[4]), to_tuple(rotated_points[5]), color=3)

cv2.line(image_1, to_tuple(rotated_points[2]), to_tuple(rotated_points[6]), color=3)

cv2.imshow('image1', image_1)
cv2.waitKey(0)

