import cv2
import numpy as np
import os
import sys

# Don't generate pyc codes
sys.dont_write_bytecode = True

class ImageUtils:
    def __init__(self):
        pass

    def get_world_coords(self, rows=9, columns=6, block_size=21.5):
        '''
        Get world coordinates for chessboard corners.
        Input:
            rows: Number of rows in chessboard
            columns: Number of columns in chessboard
            block_size: Size of each block in mm
        Output:
            world_coords: World coordinates for chessboard corners
        '''
        x, y = np.meshgrid(range(rows), range(columns))
        x, y = x.reshape(int(rows*columns), 1), y.reshape(int(rows*columns), 1)
        world_coords = np.hstack((x, y)).astype(np.float32)
        world_coords = world_coords * block_size
        return world_coords

    def get_image_coords(self, image, rows=9, columns=6, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
        '''
        Get image coordinates for chessboard corners.
        Input:
            image: Image
            rows: Number of rows in chessboard
            columns: Number of columns in chessboard
            criteria: Criteria for cornerSubPix
        Output:
            corners: Image coordinates for chessboard corners
        '''
        ret, corners = cv2.findChessboardCorners(self.gray(image), (rows, columns), None)
        corners = corners.reshape(-1, 2)
        corners = cv2.cornerSubPix(self.gray(image), corners, (11, 11), (-1, -1), criteria)
        return corners

    def gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def rgb(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def get_homography(self, img_path):
        '''
        Get homography matrix for image.
        Input:
            img_path: Image path
        Output:
            M: World coordinates for chessboard corners
            m: Image coordinates for chessboard corners
            H: Homography matrix
        '''
        M = self.get_world_coords(rows=9, columns=6, block_size=21.5)
        img = cv2.imread(img_path)
        m = self.get_image_coords(img, rows=9, columns=6)
        H, _ = cv2.findHomography(M, m)
        # A_kernel = []
        # for point in range(len(M)):
        #     x, y = M[point]
        #     u, v = m[point]
        #     A = np.array([[-x, -y, -1, 0, 0, 0, u*x, u*y, u],
        #                   [0, 0, 0, -x, -y, -1, v*x, v*y, v]])
        #     A_kernel.append(A)
        # A_kernel = np.array(A_kernel).reshape(-1, 9)
        # _, _, V = np.linalg.svd(A_kernel)
        # H = V[-1].reshape(3, 3)
        # H = H / H[-2, -2]
        return M, m, H

    def splitRT(self, RT):
        '''
        Split extrinsic parameters into rotation and translation.
        Input:
            RT: Extrinsic parameters
        Output:
            R: Rotation matrix
            t: Translation vector
        '''
        r1, r2, r3, t = RT.T
        R = np.stack((r1, r2, r3), axis=1)
        t = t.reshape(-1, 1)
        return R, t

    def rectify(self, img_paths, img_coords, _img_coords, K_init, K, kC, save_path):
        '''
        Rectify images using homography matrix.
        Input:
            img_paths: Image paths
            img_coords: Image coordinates for chessboard corners
            _img_coords: Image coordinates for chessboard corners after rectification
            save_path: Path to save rectified images
        Output:
            rectified_imgs: Rectified images
        '''
        rectified_imgs = []

        for i, (img_c, _img_c, img_path) in enumerate(zip(img_coords, _img_coords, img_paths)):
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            kC = np.array(kC)
            kC = np.array([kC[0], kC[1], 0, 0])
            # print(f'kC_shape: {kC.shape}')
            H, _ = cv2.findHomography(img_c, _img_c)
            warped_img = cv2.warpPerspective(img, H, (width, height))
            rectified_img = cv2.undistort(warped_img, K_init, kC, newCameraMatrix=K)
            rectified_imgs.append(rectified_img)

            if not os.path.exists(save_path):
                print(f"Creating folder: {save_path}")
                os.makedirs(save_path)

            rectified_folder = os.path.join(save_path, 'rectified')
            if not os.path.exists(rectified_folder):
                os.makedirs(rectified_folder)

            for m, m_ in zip(np.int32(img_c), np.int32(_img_c)):
                # cv2.circle(img, tuple(m), 5, (0, 0, 255), -1)
                cv2.drawChessboardCorners(img, (9, 6), img_c, True)
                cv2.circle(warped_img, tuple(m_), 10, (0, 0, 255), -1)

            cv2.imwrite(os.path.join(save_path, 'Img_' + str(i) + '.jpg'), img)
            cv2.imwrite(os.path.join(rectified_folder, 'rectified_Im_' + str(i) + '.jpg'), warped_img)

        return rectified_imgs
