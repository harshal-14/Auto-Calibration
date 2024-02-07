import numpy as np
import cv2
#Refactor this portion of code too
class MiscUtils:
    def __init__(self, image_utils, math_utils):
        self.image_utils = image_utils
        self.math_utils = math_utils

    def loss_fn(self, params, M_pts, m_pts, Extrinsics):
        '''
        Loss function for optimization
        Input:
            params: Intrinsic parameters
            M_pts: World coordinates for chessboard corners
            m_pts: Image coordinates for chessboard corners
            Extrinsics: Extrinsic parameters
        Output:
            error: Error for optimization
        '''
        kC = (params[-2], params[-1])
        K = self.math_utils.construct_K(params)
        error = []

        for i, RT in enumerate(Extrinsics):
            e = self.calibration_error(m_pts[i], M_pts[i], K, RT, kC, use_cv2 = False)
            error = np.hstack((error, e))
        return error

    def calibration_error(self, m_i, M_i, K, RT, kC, use_cv2):
        '''
        Calculate reprojection error.
        Input:
            m_i: Image coordinates for chessboard corners
            M_i: World coordinates for chessboard corners
            K: Intrinsic matrix
            RT: Extrinsic parameters
            kC: Distortion coefficients
            use_cv2: Use cv2 function to estimate reprojection error
        Output:
            error: Reprojection error
        '''
        R, t = self.image_utils.splitRT(RT)
        ones = np.ones(len(m_i)).reshape(-1, 1)

        if use_cv2 == False:
            m_i_ = self.math_utils.project_coords(M_i, RT, K, kC)
        else:
            kC = (kC[0], kC[1], 0, 0)
            M_i = np.column_stack((M_i, ones))
            m_i_, _ = cv2.projectPoints(M_i, R, t, K, kC)

        error = []
        for m, m_ in zip(m_i, m_i_.squeeze()):
            e_ = np.linalg.norm(m - m_, ord=2)
            error.append(e_)
        return np.sum(error)
