# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 14:25:14 2022

@author: zahra
"""

# Importing the libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal
from datetime import datetime
from scipy.ndimage.filters import maximum_filter
import imutils

# pick up from 
def pts2kps(pts, size=2): 
    kps = []
    if pts is not None: 
        #print("pts.ndim", pts.ndim)
        # convert matrix [Nx2] of pts into list of keypoints  
        kps = [ cv.KeyPoint(p[1], p[0], _size=size) for p in pts ]                      
    return kps 

def loadImage(imageDIR, imageType  = 'gray', imageScale = 'zero_TO_one'):
        I = cv.imread(imageDIR,cv.IMREAD_COLOR)
        # Convert BGR to RGB
        I = cv.cvtColor(I, cv.COLOR_BGR2RGB)
        if imageType == 'gray' :
            I = cv.cvtColor(I, cv.COLOR_BGR2GRAY)
        elif imageType == 'rgb' : 
            pass
        else : print("Unkown imageType.") 
            
        if imageScale == 'zero_TO_one' : 
            return I/255
        elif imageScale == 'zero_TO_255':
            return I
        else : print("Unkown imageScale.")
        
def rotation_Experiment(original_Image, grayScale_Image,kernel):
    Rotated_grayScale_Image = imutils.rotate(grayScale_Image, angle=45)
    Rotated_original_Image  = imutils.rotate(original_Image, angle=45)
    # define harris 
    harris = harrisFeatureDetector(kernel,k = 0.05, decisionThreshold = 0.3, nonMaximalSuppression = True, NMS_maskSize = 7 ) 
    # Original image
    pts = harris.detect(grayScale_Image)
    Ikpt = cv.drawKeypoints(original_Image, pts2kps(pts), None, color=(255,0,0))
    # Rotated Image
    pts_rotated = harris.detect(Rotated_grayScale_Image)
    Ikpt_rotated = cv.drawKeypoints(Rotated_original_Image, pts2kps(pts_rotated), None, color=(255,0,0))
    return Ikpt, Ikpt_rotated
          
       
class harrisFeatureDetector():
    def __init__(self, kernel, k, decisionThreshold, nonMaximalSuppression = False, NMS_maskSize = 0 ):
        
        """
        img - Input image. It should be grayscale and float32 type.
        blockSize - It is the size of neighbourhood considered for corner detection
        ksize - Aperture parameter of the Sobel derivative used.
        k - Harris detector free parameter in the equation.
        """
        self.kernel = kernel
        self.k = k
        self.decisionThreshold = decisionThreshold
        self.nonMaximalSuppression = nonMaximalSuppression
        self.NMS_maskSize = NMS_maskSize
        self.computTime = 0
        
        
    def set_kernel(self,kernel): self.kernel = kernel
    def set_k(self,k) : self.k = k
    def set_decisionThreshold(self, decisionThreshold) : self.decisionThreshold = decisionThreshold
    def set_nonMaximalSuppression(self,nonMaximalSuppression) : self.nonMaximalSuppression = nonMaximalSuppression
    def set_NMS_maskSize (self,NMS_maskSize) : self.NMS_maskSize = NMS_maskSize
    
    def displayComputeTime(self):
        print('Compute Time of the Harris Detector: {}'.format(self.computTime))
        
    def detect(self, grayScale_Image):
        
        start_time = datetime.now()
        # compute the gradient over the x and y axis
        Ix, Iy = np.gradient(grayScale_Image)
        Ix2 = Ix**2
        Iy2 = Iy**2
        IxIy = Ix*Iy
    
        sum_Ix2  = signal.convolve2d(Ix2, self.kernel, mode='same')
        sum_Iy2  = signal.convolve2d(Iy2, self.kernel, mode='same')
        sum_IxIy = signal.convolve2d(IxIy,self.kernel, mode='same')
        
        # Compute the Harris Criterion response
        C = sum_Ix2 * sum_Iy2 - sum_IxIy - self.k*(sum_Ix2 + sum_Iy2)**2  # #det(M) − ktrace(M)
        
        if self.nonMaximalSuppression == True:
            #% Step 4: Find local maxima (non maximum suppression)
            cNMS = C*(C == maximum_filter(C,footprint=np.ones((self.NMS_maskSize,self.NMS_maskSize)))) 
            pts = np.argwhere(cNMS>self.decisionThreshold)
        
        else : pts = np.argwhere(C>self.decisionThreshold)
        
        self.computTime = datetime.now() - start_time
        return pts
    
# ************************************************************************************************************************************* #    
class FASTFeatureDetector():
    def __init__(self,windowSize):
        
        """
        """
        self.windowSize = 12 # nombre de pixels consécutifs
        self.computTime = 0
    
    def displayComputeTime(self):
        print('Compute Time of the Harris Detector: {}'.format(self.computTime))
        
        
    def set_windowSize(self,n): self.windowSize = n
        
    def detect(self, grayScale_Image):
        
        start_time = datetime.now()
        
        yy, xx = np.meshgrid(range(3,grayScale_Image.shape[1]-3),range(3,grayScale_Image.shape[0]-3))
        XX, YY = xx.flatten()[:, np.newaxis], yy.flatten()[:, np.newaxis]  
        
        decX = np.array([-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3])
        decY = np.array([0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1])
        xN = XX + decX
        yN = YY + decY
        idxN = xN * grayScale_Image.shape[1] + yN
        
        # Récupérer l'intensité du voisinage de chaque point
        imgN = grayScale_Image.flatten()[idxN]
        # Récupérer l'intensité du voisinade de chaque image dans l'image croper des bords
        idx = XX * grayScale_Image.shape[1] + YY
        PimgFlatten = grayScale_Image.flatten()[idx]
        
        # pour chaque pixel un seuil  I > Ip0 + threshold  or I < Ip0 - threshold
        threshold = 34*np.ones((PimgFlatten.shape[0],1))
        dplus  = np.array(imgN > PimgFlatten + threshold)
        dminus = np.array(imgN < PimgFlatten - threshold)
        
        IcircleBrighter = np.array([np.where(dplus == True, 1, 0)])[0,:,:]
        IcircleDarker   = np.array([np.where(dminus == True, 1, 0)])[0,:,:]
        
        IcircleBrighter_concat = np.concatenate((IcircleBrighter,IcircleBrighter), axis = 1)
        IcircleDarker_concat   = np.concatenate((IcircleDarker,IcircleDarker), axis = 1) 
        
        B =  np.array( signal.convolve2d(IcircleBrighter_concat, np.ones((1,self.windowSize)), mode='same')) #
        D =  np.array( signal.convolve2d(IcircleDarker_concat, np.ones((1,self.windowSize)) , mode='same'))
        
        #B = np.delete(B, [0,1,2,3,4,5, 21,22,23,24,25,26], axis = 1) 
        #D = np.delete(D, [0,1,2,3,4,5, 21,22,23,24,25,26], axis = 1) 
        
        brighterPoints = np.unique(np.argwhere(B == self.windowSize)[:,0])
        darkerPoints   = np.unique( np.argwhere(D == self.windowSize)[:,0])
        
        pts1 = np.transpose(np.array( [XX[brighterPoints] , YY[brighterPoints]] ))
        pts2 = np.transpose(np.array( [XX[darkerPoints], YY[darkerPoints]] ))
        
        pts = np.concatenate((pts1,pts2), axis = 1)
        
        self.computTime = datetime.now() - start_time
        
        return pts[0,:,:]
    



