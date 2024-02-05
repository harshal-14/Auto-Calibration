import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from misc.utils import *
from scipy import optimize

def v_matrix(H, i,j):
    i,j = i-1,j-1
    v_ij = np.array([H[0, i]*H[0, j],
                    H[0, i]*H[1, j] + H[1, i]*H[0, j],
                    H[1, i]*H[1, j],
                    H[2, i]*H[0, j] + H[0, i]*H[2, j],
                    H[2, i]*H[1, j] + H[1, i]*H[2, j],
                    H[2, i]*H[2, j] 
                    ])
    return v_ij
def K_matrix(params):
    alpha, beta, gamma, u0, v0, _ , _ = params
    return np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,      1]], dtype = np.float64)
    
def IntrinsicParameters(H_matrices):
    
    """
    Compute intrinsic paramters K from given set of homographies as suggested in zhang's method 
    
    had two methods:
    method 1 -  by just substituting the values in given equation in zhang's paper
    method2 - by cholesky decomposition
    """

    # V matrix : 
    V = []
    for h in H_matrices:
        V.append(v_matrix(h, i=1, j=2)) # v12
        V.append(v_matrix(h, i=1, j=1) - v_matrix(h, i=2, j=2)) # v11-v22

    # solve V.b = 0
    U, S, Vt = np.linalg.svd(np.array(V))
    b11, b12, b22, b13, b23, b33 = Vt[np.argmin(S)]

#     # method 2:  using cholesky decomposition

#     # Construct B matrix
#     B = np.array([[b11, b12, b13],[b12, b22, b23],[b13, b23, b33]])
#     # Perform cholesky decomposition and 
#     Ktmp = np.linalg.cholesky(B)
#     K = np.linalg.inv(Ktmp).T 
#     K = K/K[2][2]
#     # enforce upper triangularity, as very low values  creep in.
#     K[1][0] , K[2][0], K[2][1] = 0, 0, 0

    # method 1 : In appendix B
    v0 = (b12*b13 - b11*b23)/(b11*b22 - b12**2)
    lamda = b33 - (b13**2 + v0*(b12*b13 - b11*b23))/b11
    alpha = np.sqrt(lamda/b11)
    beta = np.sqrt(lamda*b11 /(b11*b22 - b12**2))
    gamma = -1*b12*(alpha**2)*beta/lamda
    u0 = gamma*v0/beta -b13*(alpha**2)/lamda

    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,      1]])
    return K

def ExtrinsicParameters(K, H):
    
    """
    Compute extrinsic parameters for a given H matrix and
    intrinsic calibration parameters K
    """
    
    
    h1,h2,h3 = H.T # get the column vectors

    K_inv = np.linalg.inv(K)
    lamda = np.linalg.norm(K_inv.dot(h1),ord =2 )
    r1 = lamda*K_inv.dot(h1)
    r2 = lamda*K_inv.dot(h2)
    r3 = np.cross(r1,r2)
    t = lamda*K_inv.dot(h3)
    
    return np.stack((r1,r2,r3,t), axis=1)

def estimateReprojectionError(K,kC, data, iscv2 = False):
    M_pts, m_pts, Extrinsics = data
    errors = []
    for i,RT in enumerate(Extrinsics):
        # estimate reprojection error for all points in given image
#         e = reProjectionError(M_pts[i], m_pts[i], K, RT, kC)
        e = reProjectionError(M_pts[i], m_pts[i], K, RT, kC, iscv2 = iscv2)
        errors.append(e) 
    return np.mean(errors)

def reProjectionError(M_i, m_i, K, RT, kC, iscv2 = False):
    """
    Compute reprojection Error. 
    
    iscv2: True - runs cv2.project points to project the world points to image points
            False -  runs the self written code to perform projection
            
    m_i : image coordinates for a given image i 
    M_i : world coordinates for a given image i (same for all 13 images anyway) 
    K : intrinsic parameters
    RT : extrinsic parameters
    kC : distortion coefficients
    
    """

    """
    Compute reprojection Error.- From scratch
    works but slow, so NOT BEING USED
    
    """
    R,t = splitRT(RT)
    ones = np.ones(len(m_i)).reshape(-1,1)
    zeros=  np.zeros(len(m_i)).reshape(-1,1)

    if iscv2 == False:
        m_i_ = projectPoints(M_i, RT, K, kC)
    else:
        kC = (kC[0],kC[1], 0, 0)
        # make image and world points as homogenous coordinates
        M_i = np.column_stack((M_i,ones)) 
        ## reprojected points
        m_i_,_ = cv2.projectPoints(M_i, R, t, K, kC)

    error = []            
    for m, m_ in  zip(m_i, m_i_.squeeze()):
        e_ = np.linalg.norm(m - m_, ord=2) # compute L2 norm
        error.append(e_)
        
    return np.mean(error)

def projectPoints(M_i, RT, K, kC):
    u0,v0 = K[0,2], K[1,2]
    alpha,beta = K[0,0], K[1,1]
    k1,k2 = kC    
    m_i_ = []
    for M in M_i:
        M = np.float64(np.hstack((M,0,1))) # make image and world points as homogenous coordinates        
        # project point from world points at extrinsic level - only apply RT
        xy_ = RT.dot(M) 
        xy_ = xy_/xy_[-1]
#         xy_ = [alpha*xy_[0]/xy_[2], beta*xy_[1]/xy_[2], 1]
#        project point from extrinsic level image points to intrinsic level        
        U = K.dot(xy_)
        U = U/U[-1]

        u,v = U[0], U[1]
        x, y = xy_[0], xy_[1]

        r_sq = x**2 + y**2
        u_ = u + (u - u0)*(k1* r_sq + k2*(r_sq**2))
        v_ = v + (v - v0)*(k1* r_sq + k2*(r_sq**2))

        # obtain projected point after applying distortion kC
        m_ = np.hstack((u_,v_))
        m_i_.append(m_)
    return np.array(m_i_)
