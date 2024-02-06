import numpy as np
from scipy import optimize
import cv2
import os
from Utils.ImageUtils import *
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

class MathUtils:
    def __init__(self, image_utils):
        self.image_utils = image_utils

    def construct_v(self, H, i, j):
        i, j = i-1, j-1
        v_ij = np.array([H[0, i]*H[0, j],
                         H[0, i]*H[1, j] + H[1, i]*H[0, j],
                         H[1, i]*H[1, j],
                         H[2, i]*H[0, j] + H[0, i]*H[2, j],
                         H[2, i]*H[1, j] + H[1, i]*H[2, j],
                         H[2, i]*H[2, j]])
        return v_ij

    def construct_K(self, params):
        alpha, beta, gamma, u0, v0, _, _ = params
        return np.array([[alpha, gamma, u0],
                         [0, beta, v0],
                         [0, 0, 1]], dtype=np.float64)

    def get_intrinsic_params(self, H_all):
        V = []

        for h in H_all:
            V.append(self.construct_v(h, i=1, j=2))
            V.append(self.construct_v(h, i=1, j=1) - self.construct_v(h, i=2, j=2))

        _, S, Vt = np.linalg.svd(np.array(V))
        b11, b12, b22, b13, b23, b33 = Vt[np.argmin(S)]

        v0 = (b12*b13 - b11*b23)/(b11*b22 - b12**2)
        lambda_ = b33 - (b13**2 + v0*(b12*b13 - b11*b23))/b11
        alpha = np.sqrt(lambda_/b11)
        beta = np.sqrt(lambda_*b11/(b11*b22 - b12**2))
        gamma = -1 * (b12*(alpha**2)*beta)/lambda_
        u0 = (gamma*v0)/beta - (b13*(alpha**2))/lambda_

        K = np.array([[alpha, gamma, u0],
                      [0, beta, v0],
                      [0, 0, 1]])
        return K

    def get_extrinsic_params(self, K, H):
        h1, h2, h3 = H.T
        inv_K = np.linalg.inv(K)
        lambda_ = np.linalg.norm(inv_K.dot(h1), ord=2)
        r1 = lambda_*inv_K.dot(h1)
        r2 = lambda_*inv_K.dot(h2)
        r3 = np.cross(r1, r2)
        t = lambda_*inv_K.dot(h3)
        return np.stack((r1, r2, r3, t), axis=1)

    def project_coords(self, M_i, RT, K, kC):
        u0, v0, alpha, beta = K[0, 2], K[1, 2], K[0, 0], K[1, 1]
        k1, k2 = kC
        R, t = self.image_utils.splitRT(RT)
        m_i_ = []

        for M in M_i:
            M = np.float64(np.hstack((M, 0, 1)))
            xy_ = RT.dot(M)
            xy_ = xy_/xy_[-1]
            x, y = xy_[0], xy_[1]

            r2 = x**2 + y**2
            radial_dist = 1 + k1*r2 + k2*r2**2
            u = alpha*x*radial_dist + u0
            v = beta*y*radial_dist + v0
            m_i_.append([u, v])

        return np.array(m_i_)

    def estimate_reprojection_error(self,K,kC, data, use_cv2):
        M_pts, m_pts, Extrinsics = data
        errors = []
        for i,RT in enumerate(Extrinsics):
            e = self.reprojection_error(M_pts[i], m_pts[i], K, RT, kC, use_cv2 = use_cv2)
            errors.append(e)
        return np.mean(errors)

    def reprojection_error(self,M_i, m_i, K, RT, kC, use_cv2):
        R,t = self.image_utils.splitRT(RT)
        ones = np.ones(len(m_i)).reshape(-1,1)
        zeros=  np.zeros(len(m_i)).reshape(-1,1)

        if use_cv2 == False:
            m_i_ = self.project_coords(M_i, RT, K, kC)
        else:
            kC = (kC[0],kC[1], 0, 0)
            M_i = np.column_stack((M_i,ones)) 
            m_i_,_ = cv2.projectPoints(M_i, R, t, K, kC)

        error = []            
        for m, m_ in  zip(m_i, m_i_.squeeze()):
            e_ = np.linalg.norm(m - m_, ord=2)
            error.append(e_)
        return np.mean(error)