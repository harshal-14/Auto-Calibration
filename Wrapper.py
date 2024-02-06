import cv2
import glob
import os
import numpy as np
from scipy import optimize
from Utils.ImageUtils import *
from Utils.MathUtils import *
import argparse
import tqdm
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

class Calibration:
    def __init__(self, data_path, save_path):
        self.data_path = data_path
        self.save_path = save_path
        self.image_utils = ImageUtils()
        self.math_utils = MathUtils(self.image_utils)

    def load_images_and_homography(self):
        im_paths = sorted(glob.glob(self.data_path + '*.jpg'))
        M_pts, m_pts, H_all = [], [], []

        for im_path in im_paths:
            M, m, H = self.image_utils.get_homography(im_path)
            M_pts.append(M)
            m_pts.append(m)
            H_all.append(H)

        return M_pts, m_pts, H_all, im_paths
    
    def initial_calibration(self, H_all):
        K_init = self.math_utils.get_intrinsic_params(H_all)
        return K_init
    
    def estimate_reprojection_error(self, K, kC, M_pts, m_pts, Extrinsics, is_cv2):
        return self.math_utils.estimate_reprojection_error(K, kC, (M_pts, m_pts, Extrinsics), is_cv2)
    
    def optimize_parameters(self, init_params, M_pts, m_pts, Extrinsics):
        optimized_x = optimize.least_squares(fun=self.Loss_Fn, x0=init_params, method="lm", args=[M_pts, m_pts, Extrinsics])
        optimal_params = optimized_x.x
        return optimal_params
    
    def Loss_Fn(self, params, M_pts, m_pts, Extrinsics):
        kC = (params[-2], params[-1])
        K = self.math_utils.construct_K(params)
        error = []
        
