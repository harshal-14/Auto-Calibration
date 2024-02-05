
import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from misc.utils import  *
from misc.estimation import  *
import argparse
from scipy import optimize

def loss(params, M_pts, m_pts, Extrinsics):
    #     l2error works faster when you have 13 (no. of images) residuals.
    kC = (params[-2], params[-1])
    K = K_matrix(params)
    error = []
    for i,RT in enumerate(Extrinsics):    
        # estimate error for all points in given image
        e = geometricError(m_pts[i], M_pts[i], K, RT, kC, iscv2 = False)
        error = np.hstack((error,e))
    print(len(error))
    return error


def geometricError(m_i, M_i, K, RT, kC, iscv2 = False):
    
    """
    Compute Geometric Error. (This is just the same as projection error tbh) 
    
    iscv2: True - runs cv2.project points to project the world points to image points
            False -  runs the self written code to perform projection
            
    m_i : image coordinates for a given image i 
    M_i : world coordinates for a given image i (same for all 13 images anyway) 
    K : intrinsic parameters
    RT : extrinsic parameters
    kC : distortion coefficients
    
    """
    R,t = splitRT(RT)
    ones = np.ones(len(m_i)).reshape(-1,1)
    zeros=  np.zeros(len(m_i)).reshape(-1,1)

    if iscv2 == False:
        m_i_ = projectPoints(M_i, RT, K, kC)
    else:
        kC = (kC[0],kC[1], 0, 0)
        M_i = np.column_stack((M_i,ones)) # make image and world points as homogenous coordinates 
        m_i_, _ = cv2.projectPoints(M_i, R, t, K, kC) ## reprojected points

    error = [] 
    for m, m_ in  zip(m_i, m_i_.squeeze()):
        e_ = np.linalg.norm(m - m_, ord=2) # compute L2 norm
        error.append(e_)
#         error = np.hstack((error, e_))
    
#     return error            
    return np.sum(error) # sum the errors as given in the paper


def main():
    
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--DataPath', default="/home/lucifer/WPI/Spring_courses/CV/hbhat_hw1/Data/Calibration_Imgs/", help='Data path of images, Default: ./Calibration_Imgs/')
    Parser.add_argument('--SavePath', default='/home/lucifer/WPI/Spring_courses/CV/hbhat_hw1/Data/Outputs/', help='Path to save Results, Default: ./Outputs/')
    
    Args = Parser.parse_args()
    Savepath = Args.SavePath
    dataPath = Args.DataPath
    
    f = open("results.txt", "w")
    im_paths = sorted(glob.glob(dataPath+'*.jpg'))
    M_pts, m_pts, H_matrices = [], [], [] # obj points and image points respectively.
    for im_path in im_paths:
        M, m, H = getCorrespondences(im_path)
        M_pts.append(M)
        m_pts.append(m)    
        H_matrices.append(H)

    # Estimate K matrix - initial guess
    K_init = IntrinsicParameters(H_matrices)
    f.write('Initially estimated K matrix : \n '+ str(np.matrix.round(K_init,3))+ '\n\n')

    # Estimate Extrinsic parameters for every image
    Extrinsics_old = []
    for i, H in enumerate(H_matrices):
        RT = ExtrinsicParameters(K_init,H)
        Extrinsics_old.append(RT)    

    kC = (0,0)
    print('Distortion Coordinates before optimization:', kC)
    f.write('Distortion Coordinates before optimization: '+ str(np.round(kC,5))+ '\n')
    reprojection_error =estimateReprojectionError(K_init, kC, (M_pts, m_pts, Extrinsics_old),  False)
    reprojection_error2 =estimateReprojectionError(K_init, kC, (M_pts, m_pts, Extrinsics_old), True)
    print("Projection error before optimization, : " + str(reprojection_error2) + '(in-built)')
    print("Projection error before optimization, : " + str(reprojection_error))
    f.write('Projection error before optimization : '+ str(np.round(reprojection_error,5))+ '\n\n\n')

    print('Begin Optimization..... This takes a while')

    alpha, beta, gamma = K_init[0, 0], K_init[1, 1], K_init[0, 1]
    u0,v0 = K_init[0, 2], K_init[1, 2]
    k1,k2 = 0, 0

    init_params = [alpha, beta, gamma, u0, v0, k1, k2]
    out = optimize.least_squares(fun = loss, x0 = init_params, method="lm", args = [M_pts, m_pts, Extrinsics_old])
    optimal_params = out.x

    kC = (optimal_params[-2], optimal_params[-1])
    K = K_matrix(optimal_params)
    f.write('optimized K matrix : \n '+ str(np.matrix.round(K,3))+ '\n\n')

    # Re estimate Extrinsic parameters for every image
    Extrinsics_new = []
    m_pts_ = []
    for i, H in enumerate(H_matrices):
        RT = ExtrinsicParameters(K,H)
        Extrinsics_new.append(RT)

        M_i = np.column_stack((M_pts[i], np.ones(len(M_pts[i]))))
        R,t = splitRT(RT)
        m_i_, _ = cv2.projectPoints(M_i, R, t, K, (kC[0],kC[1], 0, 0))
        m_pts_.append(m_i_.squeeze())

    # print('Best Case setting... use l2 norm error')
    print('Distortion Coordinates after optimization: ', kC)
    f.write('Distortion Coordinates after optimization: '+ str(np.round(kC,5))+ '\n')
    
    ## to evaluate K and kC
    reprojection_error = estimateReprojectionError(K, kC, (M_pts, m_pts, Extrinsics_new),  False)
    reprojection_error2 = estimateReprojectionError(K, kC, (M_pts, m_pts, Extrinsics_new), True)
    print("Projection error after optimization, : ", reprojection_error2, '(in-built)')
    print("Projection error after optimization, : ", reprojection_error)

    f.write('Projection error after optimization : '+ str(np.round(reprojection_error,5))+ '\n\n\n')

    # Rectify on reprojected image coordinates
    Rectify( im_paths, np.array(m_pts), np.array(m_pts_), Savepath)
    
    print('Done calibration')

    f.write('##################################################################')
    f.close()

if __name__ == '__main__':
    main()
