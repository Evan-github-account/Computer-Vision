from cv2 import INTER_NEAREST
import numpy as np
import cv2


class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)

        gaussian_images=[]
        for i in range(self.num_guassian_images_per_octave):
            blur=cv2.GaussianBlur(image,(0,0),self.sigma**(i))
            gaussian_images.append(blur)       
        downsampled=cv2.resize(gaussian_images[4],(0,0),fx=0.5,fy=0.5,interpolation=INTER_NEAREST)
        #print(downsampled.shape)
        for i in range(self.num_guassian_images_per_octave):
            blur=cv2.GaussianBlur(downsampled,(0,0),self.sigma**(i))
            gaussian_images.append(blur)
        

        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)

        dog_images = []
        for i in range(self.num_DoG_images_per_octave):
            substract=cv2.subtract(gaussian_images[i+1],gaussian_images[i])
            dog_images.append(substract)
        for i in range(self.num_DoG_images_per_octave,self.num_DoG_images_per_octave*self.num_octaves):        
            substract2=cv2.subtract(gaussian_images[i+2],gaussian_images[i+1])
            dog_images.append(substract2)

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint      
        #     
        keypoints =[]
        for i in range (2,self.num_DoG_images_per_octave-1):
            for j in range (1,dog_images[i].shape[0]-1):
                for k in range (1,dog_images[i].shape[1]-1):
                    middle_point=dog_images[i][j,k]
                    kernel=[dog_images[i][j-1,k-1],dog_images[i][j-1,k],dog_images[i][j-1,k+1],dog_images[i][j,k-1],dog_images[i][j,k+1],dog_images[i][j+1,k-1],dog_images[i][j+1,k],dog_images[i][j+1,k+1],
                    dog_images[i-1][j,k],dog_images[i-1][j-1,k-1],dog_images[i-1][j-1,k],dog_images[i-1][j-1,k+1],dog_images[i-1][j,k-1],dog_images[i-1][j,k+1],dog_images[i-1][j+1,k-1],dog_images[i-1][j+1,k],dog_images[i-1][j+1,k+1],
                    dog_images[i+1][j,k],dog_images[i+1][j-1,k-1],dog_images[i+1][j-1,k],dog_images[i+1][j-1,k+1],dog_images[i+1][j,k-1],dog_images[i+1][j,k+1],dog_images[i+1][j+1,k-1],dog_images[i+1][j+1,k],dog_images[i+1][j+1,k+1]]
                    if abs(middle_point)>self.threshold:                                               
                        if middle_point>=np.max(kernel) or middle_point<=np.min(kernel):
                            keypoints.append([j,k])
                                
        for i in range (self.num_DoG_images_per_octave+2,self.num_DoG_images_per_octave*self.num_octaves-1):
            for j in range (1,dog_images[i].shape[0]-1):
                for k in range (1,dog_images[i].shape[1]-1):
                    middle_point=dog_images[i][j,k]
                    kernel=[dog_images[i][j-1,k-1],dog_images[i][j-1,k],dog_images[i][j-1,k+1],dog_images[i][j,k-1],dog_images[i][j,k+1],dog_images[i][j+1,k-1],dog_images[i][j+1,k],dog_images[i][j+1,k+1],
                    dog_images[i-1][j,k],dog_images[i-1][j-1,k-1],dog_images[i-1][j-1,k],dog_images[i-1][j-1,k+1],dog_images[i-1][j,k-1],dog_images[i-1][j,k+1],dog_images[i-1][j+1,k-1],dog_images[i-1][j+1,k],dog_images[i-1][j+1,k+1],
                    dog_images[i+1][j,k],dog_images[i+1][j-1,k-1],dog_images[i+1][j-1,k],dog_images[i+1][j-1,k+1],dog_images[i+1][j,k-1],dog_images[i+1][j,k+1],dog_images[i+1][j+1,k-1],dog_images[i+1][j+1,k],dog_images[i+1][j+1,k+1]]  
                    if abs(middle_point)>self.threshold:                      
                        if middle_point>=np.max(kernel) or middle_point<=np.min(kernel):
                            keypoints.append([j*2,k*2])                        

        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints,axis=0)

        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))] 
        return keypoints
