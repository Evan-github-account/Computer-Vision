import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)
    return img

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)
    path='./testdata/DoG1-'
    ### TODO ###
    DoG = Difference_of_Gaussian(args.threshold)
    dog_images= DoG.get_keypoints(img)
    for i in range (len(dog_images)):
        max=np.max(dog_images[i])
        min=np.min(dog_images[i])
        for j in range(dog_images[i].shape[0]):
            for k in range (dog_images[i].shape[1]):
                dog_images[i][j,k] = int(((dog_images[i][j,k]-min)/(max-min))*255)
    for i in range(len(dog_images)):
        cv2.imwrite(path+str(i+1)+'.png',dog_images[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()