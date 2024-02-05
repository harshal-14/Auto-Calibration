import numpy as np
import cv2
import copy
import glob
import argparse

class CameraCalibration:
    def __init__(self, images):
        self.images = copy.deepcopy(images)
        self.H_matrices = []
        self.camera_coordinates = []
        self.world_points = self.compute_world_points()

    def compute_world_points(self, rows=9, columns=6, block_size=21.5):
        x_world, y_world = np.meshgrid(range(rows), range(columns))
        world_points = np.array(np.vstack((x_world.flatten(), y_world.flatten())).T * block_size, dtype=np.float32)
        return world_points

    def find_homography_matrices(self, rows=9, columns=6):
        for img in self.images:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            is_corner, img_corners = cv2.findChessboardCorners(gray_img, (rows, columns), None)

            if is_corner:
                img_corners = np.reshape(img_corners, (-1, 2))
                H = self.compute_homography_matrix(self.world_points, img_corners)
                # H = cv2.findHomography(self.world_points, img_corners)[0]
                self.H_matrices.append(H)
                self.camera_coordinates.append(img_corners)
                draw = cv2.drawChessboardCorners(img, (rows, columns), img_corners, True)
                # cv2.imshow('img', draw)
                # cv2.waitKey(0)

    def construct_v(self, H, i, j):
        H = H.T
        v_ij = np.array([
            H[i][0] * H[j][0],
            H[i][0] * H[j][1] + H[i][1] * H[j][0],
            H[i][1] * H[j][1],
            H[i][2] * H[j][0] + H[i][0] * H[j][2],
            H[i][2] * H[j][1] + H[i][1] * H[j][2],
            H[i][2] * H[j][2]
        ])
        return v_ij.T

    def compute_b_matrix(self):
        V = []

        for H in self.H_matrices:
            v12 = self.construct_v(H, 0, 1).T
            v11_v22 = (self.construct_v(H, 0, 0) - self.construct_v(H, 1, 1)).T
            V.append(v12)
            V.append(v11_v22)

        V = np.array(V)
        _, _, Vh = np.linalg.svd(V, full_matrices=True)
        b = Vh.T[:, -1]

        return b
             
    def compute_homography_matrix(self, src_points, dst_points):
        A_kernel = []
        for i in range(len(src_points)):
            x, y = src_points[i]
            u, v = dst_points[i]
            A_kernel.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
            A_kernel.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        
        A_kernel = np.array(A_kernel)
        _, _, Vh = np.linalg.svd(A_kernel, full_matrices=True)
        H = Vh[-1, :].reshape((3, 3))
        
        return H / H[2, 2]

def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--Imagepath", default="/home/lucifer/WPI/Spring_courses/CV/hbhat_hw1/Data/Calibration_Imgs/")
    args= argparser.parse_args()
    path = args.path

    images = [cv2.imread(file) for file in glob.glob(path + "/*.jpg")]

    calibration = CameraCalibration(images)
    calibration.find_homography_matrices()
    b = calibration.compute_b_matrix()
    print(b)
    # A = calibration.compute_camera_intrinsics(b)

if __name__ == '__main__':
    main()
