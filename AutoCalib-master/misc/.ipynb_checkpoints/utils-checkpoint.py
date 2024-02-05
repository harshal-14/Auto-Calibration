import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def getWorldPoints(N_x = 9, N_y = 6, width = 21.5):
    """
    N_x, N_y = 9, 6 # Number of interior corners along x-axis and y-axis
    width = 21.5  # width between points
    
    """
    
    # create a grid as shown in the pdf file with meshgrid
    x, y = np.meshgrid(range(N_x), range(N_y))
    x, y = x.reshape(int(N_x*N_y), 1), y.reshape(int(N_x*N_y), 1)
    # X are real world points spaced at 21.5 cm each 
    M = np.hstack((x, y)).astype(np.float32)
    M = M* width
    
    return M

def getImagepoints(im , N_x = 9, N_y = 6, criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)):
    
    """
    N_x, N_y = 9, 6 # Number of interior corners along x-axis and y-axis
    width = 21.5  # width between points
    
    """
    # Find ChessBoard Corners
    ret, corners = cv2.findChessboardCorners(gray(im), (N_x,N_y),None)
    corners = corners.reshape(-1,2) # reshape
    corners = cv2.cornerSubPix(gray(im),corners,(11,11),(-1,-1),criteria) # refine the corners resulted.
    return corners


def getCorrespondences(im_path):

    """
    reference: https://automaticaddison.com/how-to-perform-camera-calibration-using-opencv/
    """
    M = getWorldPoints(N_x = 9, N_y = 6, width = 21.5)
    im = cv2.imread(im_path)
    # get image corner points
    m = getImagepoints(im , N_x = 9, N_y = 6)
    # find homography between world corner points and image corner points
    H,_ = cv2.findHomography(M,m)
    
    return M, m, H


def splitRT(RT):
    r1,r2,r3,t = RT.T
    R = np.stack((r1,r2,r3), axis=1)
    t = t.reshape(-1,1)
    return R,t

def gray(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
def rgb(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def Rectify(im_paths, m_pts, m_pts_, Savepath):
    rectifiedImages = []
    for i, (m_i, m_i_, im_path) in enumerate(zip(m_pts,m_pts_,im_paths)):
        im = cv2.imread(im_path)
        height, width = im.shape[:2] 

        H, _ = cv2.findHomography(m_i, m_i_)
        warpedIm = cv2.warpPerspective(im,H,(width,height))
        
        foldercheck(Savepath)
        foldercheck(Savepath+'rectified/')

        for m, m_ in zip(np.int32(m_i), np.int32(m_i_)):
            cv2.circle(im,tuple(m),15,(0,255,0),-1)
            cv2.circle(warpedIm,tuple(m_),15,(255,0,0),-1)
            
        cv2.imwrite(Savepath+'im_'+str(i)+'.jpg' , im)
        cv2.imwrite(Savepath+'rectified/'+'recIm_'+str(i)+'.jpg' , warpedIm)
            
def foldercheck(Savepath):
    if(not (os.path.isdir(Savepath))):
        print(Savepath, "  was not present, creating the folder...")
        os.makedirs(Savepath)
            

