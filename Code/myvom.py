
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage
import numpy as np
import random
from numpy.linalg import matrix_rank
import math
import cv2
import matplotlib.pyplot as plt
import os
import copy
import pandas as pd

frames = []

pathimage="/home/arjun/Desktop/VOM/Oxford_dataset/stereo/centre/"
pathmodel="/home/arjun/Desktop/VOM/Oxford_dataset/model/"


for frame in os.listdir(pathimage):
    frames.append(frame)
    frames.sort()

fx, fy, c_x, c_y, camera_img, LUT = ReadCameraModel(pathmodel)
K = np.array([[fx , 0 , c_x],[0 , fy , c_y],[0 , 0 , 1]])



def FundamentalMatrix(edge1, edge2):
    a_x = np.empty((8, 9))

    for i in range(0, len(edge1)):
        x_1 = edge1[i][0]
        y_1 = edge1[i][1]
        x_2 = edge2[i][0]
        y_2 = edge2[i][1]
        a_x[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

    u, s, v = np.linalg.svd(a_x, full_matrices=True)  
    f = v[-1].reshape(3,3)
    u1,s1,v1 = np.linalg.svd(f)
    s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]])
    F = u1 @ s2 @ v1    
    return F  

def FmatrixCondition(x1,x2,F):
    x11=np.array([x1[0],x1[1],1]).T
    x22=np.array([x2[0],x2[1],1])
    return abs(np.squeeze(np.matmul((np.matmul(x22,F)),x11)))

def EssentialMatrix(calibrationMatrix, Fmatrix):
    E = np.matmul(np.matmul(calibrationMatrix.T, Fmatrix), calibrationMatrix)
    u, s, v = np.linalg.svd(E, full_matrices=True)
    s_new = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    flag = np.matmul(u, s_new)
    e_matrix = np.matmul(flag, v)
    return e_matrix


def ObtainUniquePose(E):
    u, s, v = np.linalg.svd(e_matrix, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
   
    q1 = u[:, 2]
    r1 = u @ w @ v
   
    if np.linalg.det(r1) < 0:
        q1 = -q1
        r1 = -r1
    q1 = q1.reshape((3,1))
   
    q2 = -u[:, 2]
    r2 = u @ w @ v
    if np.linalg.det(r2) < 0:
        q2 = -q2
        r2 = -r2
    q2 = q2.reshape((3,1))
   
    q3 = u[:, 2]
    r3 = u @ w.T @ v
    if np.linalg.det(r3) < 0:
        q3 = -q3
        r3 = -r3
    q3 = q3.reshape((3,1))
   
    q4 = -u[:, 2]
    r4 = u @ w.T @ v
    if np.linalg.det(r4) < 0:
        q4 = -q4
        r4 = -r4
    q4 = q4.reshape((3,1))
   
    return [r1, r2, r3, r4], [q1, q2, q3, q4]

# def pnp(xw_new,f1,f2):
#     nul=np.zeros((1,4))
#     ii=np.array([1])
#     Xtil=np.concatenate((xw_new[0],ii))
#     Xtil=Xtil.reshape(1,4)
#     for i in range(6):
# #         u=f1[i][0]
# #         v=f1[i][1]
# #         a1=np.hstack((nul,-Xtil,v*Xtil))
# #         a2=np.hstack((Xtil,nul,-u*Xtil))
# #         a3=np.hstack((-v*Xtil,u*Xtil,nul))
# #         A1=np.vstack((a1,a2,a3))
#         u=f2[i][0]
#         v=f2[i][1]
#         a1=np.hstack((nul,-Xtil,v*Xtil))
#         a2=np.hstack((Xtil,nul,-u*Xtil))
#         a3=np.hstack((-v*Xtil,u*Xtil,nul))
#         A2=np.vstack((a1,a2,a3))
#         if i==0:
#             A=A2
#         else:
#             A=np.vstack((A,A2))
    
#     u, s, v = np.linalg.svd(A, full_matrices=True)
#     Pnew = v[-1].reshape(3,4)
#     Ro=np.linalg.inv(K)@Pnew[:,0:3]
#     u, s, v = np.linalg.svd(Ro, full_matrices=True)
#     Ro=u@v
#     T=np.linalg.inv(K)@Pnew[:,3]/s[0]
#     T=T.reshape(3,1)
#     Pnp=K@np.hstack((Ro,T))
#     return Ro,T,Pnp

# def nlt(X):
#     su=0
#     b=0
#     e=3
#     for i in range (6):
#         ii=np.array([1])
#         a=np.dot(P[0][0].T,np.concatenate((X[b:e],ii)))/np.dot(P[0][2].T,np.concatenate((X[b:e],ii)))
#         bb=np.dot(P[0][1].T,np.concatenate((X[b:e],ii)))/np.dot(P[0][2].T,np.concatenate((X[b:e],ii)))        
#         c=np.dot(P[1][1].T,np.concatenate((X[b:e],ii)))/np.dot(P[1][2].T,np.concatenate((X[b:e],ii)))        
#         d=np.dot(P[1][1].T,np.concatenate((X[b:e],ii)))/np.dot(P[1][2].T,np.concatenate((X[b:e],ii)))
#         su+= np.square (f1[i][0]-a)  + np.square (f1[i][1]-bb) + np.square (f2[i][0]-c) + np.square (f2[i][1]-d)
#         print(f1[i][0])
#         print(a)
#         b=e
#         e=e+3
#     return su


# def npnp(X):
#     su=0
# #     b=0
# #     e=3
#     for i in range (6):
#         ii=np.array([1])
#         a=np.dot(P[0][0].T,np.concatenate((xw_new[i],ii)))/np.dot(P[0][2].T,np.concatenate((xw_new[i],ii)))
#         bb=np.dot(P[0][1].T,np.concatenate((xw_new[i],ii)))/np.dot(P[0][2].T,np.concatenate((xw_new[i],ii)))        
# #         c=np.dot(P[1][1].T,np.concatenate((X[b:e],ii)))/np.dot(P[1][2].T,np.concatenate((X[b:e],ii)))        
# #         d=np.dot(P[1][1].T,np.concatenate((X[b:e],ii)))/np.dot(P[1][2].T,np.concatenate((X[b:e],ii)))
#         su+= np.square (f1[i][0]-a)  + np.square (f1[i][1]-bb) 
# #         print(f1[i][0])
# #         print(a)
#         b=e
#         e=e+3
#     return su

def ObtainEulerAngles(rot_mat) :
    eu1er = math.sqrt(rot_mat[0,0] * rot_mat[0,0] +  rot_mat[1,0] * rot_mat[1,0])
    singular_value = eu1er < 1e-6
 
    if  not singular_value :
        x = math.atan2(rot_mat[2,1] , rot_mat[2,2])
        y = math.atan2(-rot_mat[2,0], eu1er)
        z = math.atan2(rot_mat[1,0], rot_mat[0,0])

    else :
        x = math.atan2(-rot_mat[1,2], rot_mat[1,1])
        y = math.atan2(-rot_mat[2,0], eu1er)
        z = 0
    return np.array([x*180/math.pi, y*180/math.pi, z*180/math.pi])

def HomogenousMatrix(rot_mat, t):
    i = np.column_stack((rot_mat, t))
    a = np.array([0, 0, 0, 1])
    H = np.vstack((i, a))
    return H

def GetLinearTriangulationPoint(m1, m2, point1, point2):
    old_x = np.array([[0, -1, point1[1]], [1, 0, -point1[0]], [-point1[1], point1[0], 0]])
    old_xdash = np.array([[0, -1, point2[1]], [1, 0, -point2[0]], [-point2[1], point2[0], 0]])
    a1 = old_x @ m1[0:3, :]
    a2 = old_xdash @ m2
    a_x = np.vstack((a1, a2))
    u, s, v = np.linalg.svd(a_x)
    new1X = v[-1]
    new1X = new1X/new1X[3]
    new1X = new1X.reshape((4,1))
    return new1X[0:3].reshape((3,1))

def UniqueCameraPose(RotationMatrix, CameraCenter, features1, features2):
    ispoint3D = 0
    Horigin = np.identity(4)
    for index in range(0, len(RotationMatrix)):
        angles = ObtainEulerAngles(RotationMatrix[index])
        if angles[0] < 50 and angles[0] > -50 and angles[2] < 50 and angles[2] > -50:
            count = 0
            newP = np.hstack((RotationMatrix[index], CameraCenter[index]))
            for i in range(0, len(features1)):
                temp1x = GetLinearTriangulationPoint(Horigin[0:3,:], newP, features1[i], features2[i])
                thirdrow = RotationMatrix[index][2,:].reshape((1,3))
                if np.squeeze(thirdrow @ (temp1x - CameraCenter[index])) > 0:
                    count = count + 1
            if count > ispoint3D:
                ispoint3D = count
                Translation_final = CameraCenter[index]
                Rotation_final = RotationMatrix[index]
               
    if Translation_final[2] > 0:
        Translation_final = -Translation_final
               
    return Rotation_final, Translation_final
   
H_Start = np.identity(4)
p_0 = np.array([[0, 0, 0, 1]]).T
flag = 0

data_points = []
for index in range(18, len(frames)-1):
    print(frames[index], index)
    img1 = cv2.imread(pathimage + str(frames[index]), 0)
    distorted_image1 = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
    undistorted_image1 = UndistortImage(distorted_image1,LUT)  
    grayscale_1 = cv2.cvtColor(undistorted_image1,cv2.COLOR_BGR2GRAY)
   
    img2 = cv2.imread(pathimage+ str(frames[index + 1]), 0)
    distorted_image2 = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
    undistorted_image2 = UndistortImage(distorted_image2,LUT)  
    grayscale_2 = cv2.cvtColor(undistorted_image2,cv2.COLOR_BGR2GRAY)

    frame_image1 = grayscale_1[200:650, 0:1280]
    frame_image2 = grayscale_2[200:650, 0:1280]

    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptor_1 = sift.detectAndCompute(frame_image1,None)
    keypoints_2, descriptor_2 = sift.detectAndCompute(frame_image2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    feature_matches = flann.knnMatch(descriptor_1,descriptor_2,k=2)
   
    srcfeatures = []
    dstfeatures = []

    for i,(m,n) in enumerate(feature_matches):
        if m.distance < 0.5*n.distance:
            srcfeatures.append(keypoints_1[m.queryIdx].pt)
            dstfeatures.append(keypoints_2[m.trainIdx].pt)
       
    Total_inliers = 0
    FinalFundamentalMatrix = np.zeros((3,3))
    inlier1 = []
    inlier2 = []
    for i in range(0, 50):
        count = 0
        Extracted_points = []
        Frame1_features = []
        Frame2_features = []
        TemporaryFeatures_1 = []
        TemporaryFeatures_2 = []
       
        while(True):
            num = random.randint(0, len(srcfeatures)-1)
            if num not in Extracted_points:
                Extracted_points.append(num)
            if len(Extracted_points) == 8:
                break

        for point in Extracted_points:
            Frame1_features.append([srcfeatures[point][0], srcfeatures[point][1]])
            Frame2_features.append([dstfeatures[point][0], dstfeatures[point][1]])
   
        FundMatrix = FundamentalMatrix(Frame1_features, Frame2_features)

        for number in range(0, len(srcfeatures)):
            if FmatrixCondition(srcfeatures[number], dstfeatures[number], FundMatrix) < 0.01:
                count = count + 1
                TemporaryFeatures_1.append(srcfeatures[number])
                TemporaryFeatures_2.append(dstfeatures[number])

        if count > Total_inliers:
            Total_inliers = count
            FinalFundamentalMatrix = FundMatrix
            inlier1 = TemporaryFeatures_1
            inlier2 = TemporaryFeatures_2
   
    e_matrix = EssentialMatrix(K, FinalFundamentalMatrix)

    RotationMatrix, Tlist = ObtainUniquePose(e_matrix)
    rot_mat, T = UniqueCameraPose(RotationMatrix, Tlist, inlier1, inlier2)

    H_Start = H_Start @ HomogenousMatrix(rot_mat, T)
    p_projection = H_Start @ p_0

    print('x- ', p_projection[0])
    print('y- ', p_projection[2])
    data_points.append([p_projection[0][0], -p_projection[2][0]])
    plt.scatter(p_projection[0][0], -p_projection[2][0], color='r')
    # s=np.array(xw)
    # s=s.reshape(s.shape[0]*s.shape[1])
    # sin=s[0:18]
    # nlt(sin)
    # res_1 = least_squares(nlt, sin)
    # ss=res_1.x
    # xw_new=ss.reshape(int(ss.shape[0]/3),3)
   
    if cv2.waitKey(0) == 27:
        break
    flag = flag + 1

cv2.destroyAllWindows()
df = pd.DataFrame(data_points, columns = ['X', 'Y'])
df.to_excel('mycoordinates.xlsx')
plt.savefig('mypath.png')
plt.show()
	
	
