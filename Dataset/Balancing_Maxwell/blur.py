import cv2
import numpy as np

def readme():
    print('Use check_blur() to see the effect. Press any key to close the image window. KernelSize is the strength of the blure. Angle is CCW rotation from horizontal in degree.')

def apply_blur(image_path, kernelSize, angle):
    #read image
    image = cv2.imread(image_path)

    #First generate a horizontal motion blur kernel
    kernel = np.zeros((kernelSize, kernelSize))
    kernel[int((kernelSize - 1) / 2), :] = np.ones(kernelSize)
    kernel = kernel / kernelSize

    #Apply rotationi
    center = (kernelSize / 2, kernelSize / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernelSize, kernelSize)) #apply the rotation matrix on the horizontal blur kernel

    return cv2.filter2D(image, -1, kernel)

def check_blur(image_path, kernelSize, angle):
    #read image
    image = cv2.imread(image_path)

    #First generate a horizontal motion blur kernel
    kernel = np.zeros((kernelSize, kernelSize))
    kernel[int((kernelSize - 1) / 2), :] = np.ones(kernelSize)
    kernel = kernel / kernelSize

    #Apply rotationi
    center = (kernelSize / 2, kernelSize / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    kernel = cv2.warpAffine(kernel, rotation_matrix, (kernelSize, kernelSize)) #apply the rotation matrix on the horizontal blur kernel

    blurred_image = cv2.filter2D(image, -1, kernel)

    #Check Images
    cv2.imshow('Original Image', image)
    cv2.imshow('Blured Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()