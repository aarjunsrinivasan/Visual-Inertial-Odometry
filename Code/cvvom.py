import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import copy
from numpy.linalg import matrix_rank
import pandas as pd

l = []
frames = []
pathimage="/home/arjun/Desktop/VOM/Oxford_dataset/stereo/centre/"
pathmodel="/home/arjun/Desktop/VOM/Oxford_dataset/model/"

def cvfunctions(distort1, distort2, k):
    kp1, des1 = sift.detectAndCompute(distort1, None)
    kp2, des2 = sift.detectAndCompute(distort2, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    pointsmatched = []
    pointsfrom1 = []
    pointsfrom2 = []

    for i, (s, p) in enumerate(matches):
        if s.distance < 0.5 * p.distance:
            pointsmatched.append(s)
            pointsfrom2.append(kp2[s.trainIdx].pt)
            pointsfrom1.append(kp1[s.queryIdx].pt)

    pointsfrom1 = np.int32(pointsfrom1)
    pointsfrom2 = np.int32(pointsfrom2)
    F, mask = cv2.findFundamentalMat(pointsfrom1, pointsfrom2, cv2.FM_RANSAC)

    pointsfrom1 = pointsfrom1[mask.ravel() == 1]
    pointsfrom2 = pointsfrom2[mask.ravel() == 1]
    
    E = k.T @ F @ k
    retval, R, t, mask = cv2.recoverPose(E, pointsfrom1, pointsfrom2, k)
    return R, t


def cameraMatrix(file):
    frames1 = []
    for frames in os.listdir(file):
        frames1.append(frames)
    fx, fy, cx, cy, G_camera_frames, LUT = ReadCameraModel(pathmodel)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return K, LUT

def Homogenousmatrix(R, t):
    h = np.column_stack((R, t))
    a = np.array([0, 0, 0, 1])
    h = np.vstack((h, a))
    return h


sift = cv2.xfeatures2d.SIFT_create()
# pathimage="/home/arjun/Desktop/VOM/Oxford_dataset/stereo/centre/"
# pathmodel="/home/arjun/Desktop/VOM/Oxford_dataset/model/"
file = pathimage
k, LUT = cameraMatrix(pathimage)

for frames1 in os.listdir(file):
    frames.append(frames1)
frames.sort()
homo1 = np.identity(4)
t1 = np.array([[0, 0, 0, 1]])
t1 = t1.T

for index in range(19, len(frames)-1):
    print(frames[index], index)
    img1 = cv2.imread(pathimage + str(frames[index]), 0)
    colorimage1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
    undistortedimage1 = UndistortImage(colorimage1,LUT)  
    gray1 = cv2.cvtColor(undistortedimage1,cv2.COLOR_BGR2GRAY)
    
    img2 = cv2.imread(pathimage+ str(frames[index + 1]), 0)
    colorimage2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
    undistortedimage2 = UndistortImage(colorimage2,LUT)  
    gray2 = cv2.cvtColor(undistortedimage2,cv2.COLOR_BGR2GRAY)

    gray1 = gray1[200:650, 0:1280]
    gray2 = gray2[200:650, 0:1280]

    R, T = cvfunctions(gray1, gray2, k)

    homo2 = Homogenousmatrix(R, T)
    homo1 = homo1 @ homo2
    p1 = homo1 @ t1

    plt.scatter(p1[0][0], -p1[2][0], color='r')
    l.append([p1[0][0], -p1[2][0]])

df = pd.DataFrame(l, columns = ['X', 'Y'])
df.to_excel('ocvcoordinates.xlsx')

plt.savefig('ocv.png')
plt.show()
