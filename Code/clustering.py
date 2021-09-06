import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from PIL import Image

def to_pil(img):
  ''' Transforms a 3 dimentional matrix into a PIL image '''
  return Image.fromarray(img.astype('uint8'), 'RGB')

def to_cv2(img):
    open_cv_image = np.array(img) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1].copy() 

def binary_to_rgb(arr):
  ''' Transforms a binary image into a RGB image '''
  arr *= 255
  return np.repeat(arr[:, :, np.newaxis], 3, axis=2)

def store_images(original,clustered):
    ''' Converts and Stores the images locally '''
    (to_pil(original)).save("Original.png")
    (to_pil(clustered)).save("Cluster.png")

def run_clustering(file_name):
    ''' Run the clustering algorithm, requires the name of the image to be opened, returns the clustered image '''
    img = cv2.imread(file_name) 
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.cvtColor(labimg, cv2.COLOR_GRAY2BGR)
    labimg = cv2.cvtColor(img_grey, cv2.COLOR_BGR2LAB)

    n = 0
    while(n<0): # change to other values for less downscale
        labimg = cv2.pyrDown(labimg)
        n = n+1
    
    rows, cols, chs = labimg.shape
    
    # A higher eps means more changes are detected.
    db = DBSCAN(eps=1, min_samples=4, metric = 'euclidean',algorithm ='auto')
    
    indices = np.dstack(np.indices(labimg.shape[:2]))
    xycolors = np.concatenate((labimg, indices), axis=-1) 
    feature_image = np.reshape(xycolors, [-1,5])
    db.fit(feature_image)
    labels = db.labels_
    
    labels[labels < 0.5] = 0  # set pixels with value < threshold to 0 
    labels[labels >= 0.5] = 1 # set pixels with value >= threshold to 1 
    
    img_cluster = np.reshape(labels, [rows, cols])
    img_cluster = binary_to_rgb(img_cluster)

    #fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    #ax[0].imshow(img)
    #ax[1].imshow(img_cluster)

    #Store the images
    #store_images(img,img_cluster)
    return img_cluster

def run_clustering_image_cv2(cv2_image):
    ''' Run the clustering algorithm, requires a cv2 image, returns the clustered image '''
    img = cv2_image
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.cvtColor(labimg, cv2.COLOR_GRAY2BGR)
    labimg = cv2.cvtColor(img_grey, cv2.COLOR_BGR2LAB)

    n = 0
    while(n<0): # change to other values for less downscale
        labimg = cv2.pyrDown(labimg)
        n = n+1
    
    rows, cols, chs = labimg.shape
    
    # A higher eps means more changes are detected.
    db = DBSCAN(eps=1, min_samples=4, metric = 'euclidean',algorithm ='auto')
    
    indices = np.dstack(np.indices(labimg.shape[:2]))
    xycolors = np.concatenate((labimg, indices), axis=-1) 
    feature_image = np.reshape(xycolors, [-1,5])
    db.fit(feature_image)
    labels = db.labels_
    
    labels[labels < 0.5] = 0  # set pixels with value < threshold to 0 
    labels[labels >= 0.5] = 1 # set pixels with value >= threshold to 1 
    
    img_cluster = np.reshape(labels, [rows, cols])
    img_cluster = binary_to_rgb(img_cluster)

    #fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    #ax[0].imshow(img)
    #ax[1].imshow(img_cluster)
    
    #Store the images
    #store_images(img,img_cluster)
    return img_cluster
    
def run_clustering_image_pil(pil_image):
    ''' Run the clustering algorithm, requires a PIL image, returns the clustered image '''
    img = to_cv2(pil_image)
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey = cv2.cvtColor(labimg, cv2.COLOR_GRAY2BGR)
    labimg = cv2.cvtColor(img_grey, cv2.COLOR_BGR2LAB)

    n = 0
    while(n<0): # change to other values for less downscale
        labimg = cv2.pyrDown(labimg)
        n = n+1
    
    rows, cols, chs = labimg.shape
    
    # A higher eps means more changes are detected.
    db = DBSCAN(eps=1, min_samples=4, metric = 'euclidean',algorithm ='auto')
    
    indices = np.dstack(np.indices(labimg.shape[:2]))
    xycolors = np.concatenate((labimg, indices), axis=-1) 
    feature_image = np.reshape(xycolors, [-1,5])
    db.fit(feature_image)
    labels = db.labels_
    
    labels[labels < 0.5] = 0  # set pixels with value < threshold to 0 
    labels[labels >= 0.5] = 1 # set pixels with value >= threshold to 1 
    
    img_cluster = np.reshape(labels, [rows, cols])
    img_cluster = binary_to_rgb(img_cluster)

    #fig, ax = plt.subplots(1, 2, figsize=(20, 20))
    #ax[0].imshow(img)
    #ax[1].imshow(img_cluster)
    
    #Store the images
    #store_images(img,img_cluster)
    return img_cluster