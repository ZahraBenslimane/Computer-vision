# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:51:16 2022

@author: zahra
"""
# Importing the libraries
import numpy as np

def getDescriptors(grayScale_Image, pts, descriptorSize):
    
    ll = np.arange(- int(np.floor(descriptorSize/2)) , int(np.floor(descriptorSize/2)+1), 1)
    xdec = np.array([[x]*descriptorSize for x in ll ]).flatten()
    ydec = np.array([ll]*descriptorSize).flatten()
    
    ptsX = pts[:,0].reshape((pts[:,0].shape[0],1))
    ptsX_Voisins = np.hstack([ptsX]*xdec.shape[0])
    indX = ptsX_Voisins - np.flip(xdec)
    
    ptsY = pts[:,1].reshape((pts[:,1].shape[0],1))
    ptsY_Voisins = np.hstack([ptsY]*ydec.shape[0])
    indY = ptsY_Voisins - np.flip(ydec)
    
    ndgImg = []
    for pX, pY in zip(indX,indY):
        ndg_voisins = []
        for i,j in zip(pX, pY):
            if  i >= grayScale_Image.shape[0] or j >= grayScale_Image.shape[1]:
                ndg_voisins.append(0)
            else :
                ndg_voisins.append(grayScale_Image[i,j])
                         
        ndgImg.append(ndg_voisins)
        
    return ndgImg


def findMatch(array,Matrix,distanceType):
    
    valid = True
    
    if distanceType == "Sum_of_Squared_Distances" :
        squared_distance = ((np.array(Matrix) - np.array(array))**2 )
        #squared_distance = squared_distance.reshape((squared_distance.shape[0],squared_distance.shape[2]))
        Sum_of_Squared_Distances = np.sum(squared_distance, axis=1)
        argMin = np.argmin(Sum_of_Squared_Distances)

    elif distanceType == "Sum_Of_Relative_Distances": 
        a = np.abs(( np.array(Matrix) - np.array(array)))
        b = ( np.array(Matrix) + np.array(array))
        relativeDistance = ((a/b)**2)
        
        #relativeDistance = relativeDistance.reshape((relativeDistance.shape[0],relativeDistance.shape[2]))
        Sum_of_relativeDistance = np.sum(relativeDistance, axis=1)
        argMin = np.argmin(Sum_of_relativeDistance)   
        
        if Sum_of_relativeDistance[argMin] <= 9*0.05:
            valid = False
        
    return argMin, valid


def featuresMatcher(grayScale_I1, pts1, grayScale_I2, pts2, descriptorSize, distanceType, TranslationOnly = True):
        
    # IMAGE 1 
    ndgImg1 = getDescriptors(grayScale_I1, pts1, descriptorSize)    
    # IMAGE 2 
    ndgImg2 = getDescriptors(grayScale_I2, pts2, descriptorSize)
    
    index_Matches = []
    myPairs = []
    for index_point_interet, ndg_voisinage in enumerate(ndgImg1):
        
        if distanceType == "Sum_of_Squared_Distances" :
            argMin1, valid1 = findMatch(ndg_voisinage,ndgImg2,distanceType)
            argMin2, valid2 = findMatch(ndgImg2[argMin1],ndgImg1,distanceType)
            if argMin2 == index_point_interet and valid1 and valid2  :
                index_Matches.append([index_point_interet, argMin1])
                
                if  TranslationOnly  and np.abs(pts1[index_point_interet][0] - pts2[argMin1][0]) < 30 : 
                    myPairs.append([pts1[index_point_interet], pts2[argMin1]])
                elif  TranslationOnly == False :  myPairs.append([pts1[index_point_interet], pts2[argMin1]])   
            
        elif distanceType == "Sum_Of_Relative_Distances": 
            argMin1, valid1 = findMatch(ndg_voisinage,ndgImg2,distanceType)
            argMin2, valid2 = findMatch(ndgImg2[argMin1],ndgImg1,distanceType)
            if argMin2 == index_point_interet and valid1 and valid2 :
                index_Matches.append([index_point_interet, argMin1])
                
                if  TranslationOnly and  np.abs(pts1[index_point_interet][0] - pts2[argMin1][0]) < 30 : 
                    myPairs.append([pts1[index_point_interet], pts2[argMin1]])
                elif  TranslationOnly == False :  myPairs.append([pts1[index_point_interet], pts2[argMin1]])   
                
        else : print("Unkown distanceType.")  
        #break   
        
        
    xyA, xyB = [],[]
    for pair in myPairs : 
        xyA.append((pair[0][0], pair[0][1]))
        xyB.append((pair[1][0], pair[1][1]))
        
    return xyA,xyB
