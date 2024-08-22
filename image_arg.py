import cv2
import numpy as np

def apply_blur(image_path, kernelSize, angle):

    #Convert values to sutible data type
    kernelSize = int(kernelSize)
    angle = int(angle)

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

def apply_rotation(image_path, angle, argPar2):

    #Convert values to sutible data type
    angle = int(angle)

    #Read image
    image = cv2.imread(image_path)

    #Generate rotational matrix
    (h, w) = image.shape[:2] #This can be fixed to increase speed
    center = (w // 2, h // 2) #First weith then hight

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    #Apply rotational matrix
    return(cv2.warpAffine(image, rotation_matrix, (w, h)))

def apply_flip(image_path, dir, argPar2):

    #Read image
    image = cv2.imread(image_path)

    #Flipe
    if dir == 'v':
        return(cv2.flip(image, 0))
    elif dir == 'h':
        return(cv2.flip(image, 1))
    else:
        raise ValueError('Unexpected flipe direction use v or h')

def apply_noise(image_path, intensity, argPar2):

    #Convert values to sutible data type
    intensity = float(intensity)

    #Read image
    image = cv2.imread(image_path)

    #Generate noise mask
    noiseMask = np.random.normal(0, intensity, image.shape).astype('uint8') #Check if the formate is unit8

    #Apply noise mask
    return(cv2.add(image, noiseMask))

def apply_briten(image_path, intensity, argPar2):

    #Convert values to sutible data type
    intensity = int(intensity)

    #Read image
    image = cv2.imread(image_path)

    #Convert image to HSV color space (so that only need to changge one value)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Splite the channels and modify the v channel for changing brightness
    h, s, v = cv2.split(hsv_image)
    v = cv2.add(v, intensity)
    mod_hsv_image = cv2.merge((h, s, v))

    #Conver the image back to RGB color space
    return(cv2.cvtColor(mod_hsv_image, cv2.COLOR_HSV2BGR))

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

def readme():
    print('Use check_blur() to see the effect. Press any key to close the image window. KernelSize is the strength of the blure. Angle is CCW rotation from horizontal in degree.')