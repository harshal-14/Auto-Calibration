import cv2
import glob
import os
import numpy as np
from scipy import optimize
from Utils.ImageUtils import *
from Utils.MathUtils import *
from Utils.MiscUtils import *
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
        self.misc_utils = MiscUtils(self.image_utils, self.math_utils)

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

    def estimate_reprojection_error(self, K, kC, M_pts, m_pts, Extrinsics, use_cv2):
        return self.math_utils.estimate_reprojection_error(K, kC, (M_pts, m_pts, Extrinsics), use_cv2)

    def optimize_parameters(self, init_params, M_pts, m_pts, Extrinsics):
        out = optimize.least_squares(fun=self.misc_utils.loss_fn, x0=init_params, method="lm", args=[M_pts, m_pts, Extrinsics])
        optimal_params = out.x
        return optimal_params

    def loss_fn(self, params, M_pts, m_pts, Extrinsics):
        return self.misc_utils.loss_fn(params, M_pts, m_pts, Extrinsics)
    
    def calibration_error(self, m_i, M_i, K, RT, kC, use_cv2=False):
        return self.misc_utils.calibration_error(m_i, M_i, K, RT, kC, use_cv2)

    def main(self):
        M_pts, m_pts, H_all, im_paths = self.load_images_and_homography()
        K_init = self.initial_calibration(H_all)

        Extrinsics_old = []
        for i, H in enumerate(H_all):
            RT = self.math_utils.get_extrinsic_params(K_init, H)
            Extrinsics_old.append(RT)

        kC = (0, 0)
        print(f'Distortion Coordinates(kC) before optimization: {kC}')
        reprojection_error = self.estimate_reprojection_error(K_init, kC, M_pts, m_pts, Extrinsics_old, False)
        print(f'Projection error before optimization: {reprojection_error}', '(without using cv2 function)')
        print(f'Initiating Optimization...')
        print(f'K before optimization: {K_init}')

        alpha, beta, gamma = K_init[0, 0], K_init[1, 1], K_init[0, 1]
        u0, v0 = K_init[0, 2], K_init[1, 2]
        k1, k2 = 0, 0
        init_params = [alpha, beta, gamma, u0, v0, k1, k2]
        
        with tqdm.tqdm(total=1, desc = 'Optimizing') as pbar:
            # for i in range(len(M_pts)):
                optimal_params = self.optimize_parameters(init_params, M_pts, m_pts, Extrinsics_old)
                pbar.update(1)

        kC = (optimal_params[-2], optimal_params[-1])
        K = self.math_utils.construct_K(optimal_params)

        Extrinsics_new = []
        m_pts_ = []

        for i, H in enumerate(H_all):
            RT = self.math_utils.get_extrinsic_params(K, H)
            Extrinsics_new.append(RT)

            M_i = np.column_stack((M_pts[i], np.ones(len(M_pts[i]))))
            R, t = self.image_utils.splitRT(RT)
            m_i_, _ = cv2.projectPoints(M_i, R, t, K, (kC[0], kC[1], 0, 0))
            m_pts_.append(m_i_.squeeze())    

        print('Distortion Coordinates after optimization:', kC)
        reprojection_error = self.estimate_reprojection_error(K, kC, M_pts, m_pts, Extrinsics_new, False)
        # reprojection_error2 = self.estimate_reprojection_error(K, kC, M_pts, m_pts, Extrinsics_new, True)
        # print("Projection error after optimization:", reprojection_error2, "(in-built)")
        # print("Projection error after optimization:", reprojection_error)
        print(f'Projection error after optimization: {reprojection_error}', '(without cv2)')
        print(f'K after optimization: {K}')
        self.image_utils.rectify(im_paths, np.array(m_pts), np.array(m_pts_), self.save_path)
        print(f'Done! Rectified images saved at: {self.save_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--InputDataPath', default="/home/lucifer/WPI/Spring_courses/CV/hbhat_hw1/Data/Calibration_Imgs/", help='Data path of images')
    parser.add_argument('--OutputPath', default='/home/lucifer/WPI/Spring_courses/CV/hbhat_hw1/Data/Outputs/', help='Path to save Results')
    
    args = parser.parse_args()
    
    calibration = Calibration(args.InputDataPath, args.OutputPath)
    calibration.main()
