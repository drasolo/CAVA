import sys
import cv2 as cv
import numpy as np
import copy

from parameters import *
from select_path import *

import pdb

def compute_energy(img):
    """
    calculeaza energia la fiecare pixel pe baza gradientului
    :param img: imaginea initiala
    :return:E - energia
    """
    # urmati urmatorii pasi:
    # 1. transformati imagine in grayscale
    # 2. folositi filtru sobel pentru a calcula gradientul in directia X si Y
    # 3. calculati magnitudinea pentru fiecare pixel al imaginii
    E = np.zeros((img.shape[0],img.shape[1]))
    #TODO: scrieti codul
    img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    grad_x=cv.Sobel(img_gray,ddepth=cv.CV_16S,dx=1,dy=0,borderType=cv.BORDER_CONSTANT)
    grad_y=cv.Sobel(img_gray,ddepth=cv.CV_16S,dx=0,dy=1,borderType=cv.BORDER_CONSTANT)

    abs_x=np.abs(grad_x)
    abs_y=np.abs(grad_y)
    E=abs_x+abs_y

    return E

def show_path(img, path, color):
    new_image = img.copy()
    for row, col in path:
        new_image[row, col] = color

    E = compute_energy(img)
    new_image_E = img.copy()
    new_image_E[:,:,0] = E.copy()
    new_image_E[:,:,1] = E.copy()
    new_image_E[:,:,2] = E.copy()

    for row, col in path:
        new_image_E[row, col] = color
    cv.imshow('path img', np.uint8(new_image))
    cv.imshow('path E', np.uint8(new_image_E))
    cv.waitKey(1000)


def delete_path(img, path):
    """
    elimina drumul vertical din imagine
    :param img: imaginea initiala
    :path - drumul vertical
    return: updated_img - imaginea initiala din care s-a eliminat drumul vertical
    """
    updated_img = np.zeros((img.shape[0], img.shape[1] - 1, img.shape[2]), np.uint8)
    for i in range(img.shape[0]):
        col=path[i][1]
        updated_img[i,:col]=img[i,:col].copy()
        updated_img[i,col:]=img[i,col+1:].copy()
    return updated_img

def decrease_width(params: Parameters, num_pixels):
    img = params.image.copy() # copiaza imaginea originala
    for i in range(num_pixels):
        print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i+1, num_pixels))

        # calculeaza energia dupa ecuatia (1) din articol
        E = compute_energy(img)
        path = select_path(E, params.method_select_path)
        if params.show_path:
            show_path(img, path, params.color_path)
        img = delete_path(img, path)

    cv.destroyAllWindows()
    return img

def decrease_height(params: Parameters, num_pixels):

    params.image=np.rot90(params.image,k=3)
    resized_image_rot=decrease_width(params,params.num_pixel_height)
    params.image=np.rot90(params.image,k=1)
    resized_image=np.rot90(resized_image_rot,k=1)

    return resized_image

def amplify_content(params:Parameters):
    original_image=params.image.copy()
    print(original_image.shape)
    scaled_image=cv.resize(params.image,(0,0),fx=params.factor_amplification,fy=params.factor_amplification)
    print(scaled_image.shape)
    params.image=scaled_image
    params.num_pixels_width=scaled_image.shape[1]-original_image.shape[1]
    print(params.num_pixels_width)
    resized_image=decrease_width(params,params.num_pixels_width)
    params.num_pixel_height=scaled_image.shape[0]-original_image.shape[0]
    print(params.num_pixel_height)
    params.image=resized_image
    resized_image=decrease_height(params,params.num_pixel_height)
    params.image=original_image
    return resized_image
def delete_object(params: Parameters, x0, y0, w, h):
    img=params.image.copy()
    if w<h:
        num_pixels=w
        for i in range(num_pixels):
            print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels))
            E=compute_energy(img)
            E2=np.zeros(E.shape)
            E2[y0:y0+h,x0:x0+w]=-100000
            E=E+E2
            w=w-1
            path=select_path(E,params.method_select_path)
            if params.show_path:
                show_path(img, path, params.color_path)
            img=delete_path(img,path)
    else:
        num_pixels=h
        img=np.rot90(img,k=3)
        for i in range(num_pixels):
            print('Eliminam drumul vertical numarul %i dintr-un total de %d.' % (i + 1, num_pixels))
            E = compute_energy(img)
            E2=np.zeros((E.shape[1],E.shape[0]))
            E2[y0:y0 + h, x0:x0 + w] = -100000
            E2=np.rot90(E2,k=3)
            E=E+E2
            h=h-1
            path = select_path(E, params.method_select_path)
            if params.show_path:
                show_path(img, path, params.color_path)
            img = delete_path(img, path)
        img=np.rot90(img,k=1)
    cv.destroyAllWindows()
    return img

def resize_image(params: Parameters):

    if params.resize_option == 'micsoreazaLatime':
        # redimensioneaza imaginea pe latime
        resized_image = decrease_width(params, params.num_pixels_width)
        return resized_image

    elif params.resize_option == 'micsoreazaInaltime':
        resized_image=decrease_height(params,params.num_pixel_height)
        return resized_image
    
    elif params.resize_option == 'amplificaContinut':
        resized_image=amplify_content(params)
        return resized_image

    elif params.resize_option == 'eliminaObiect':
        x0,y0,w,h=cv.selectROI(np.uint8(params.image))
        print(x0,y0,w,h)
        return delete_object(params,x0,y0,w,h)

    else:
        print('The option is not valid!')
        sys.exit(-1)

#===select_path.py===

import sys
import numpy as np
import pdb


def select_random_path(E):
    # pentru linia 0 alegem primul pixel in mod aleator
    line = 0
    col = np.random.randint(low=0, high=E.shape[1], size=1)[0]
    path = [(line, col)]
    for i in range(1,E.shape[0]):
        # alege urmatorul pixel pe baza vecinilor
        line = i
        # coloana depinde de coloana pixelului anterior
        if path[-1][1] == 0:  # pixelul este localizat la marginea din stanga
            opt = np.random.randint(low=0, high=2, size=1)[0]
        elif path[-1][1] == E.shape[1] - 1:  # pixelul este la marginea din dreapta
            opt = np.random.randint(low=-1, high=1, size=1)[0]
        else:
            opt = np.random.randint(low=-1, high=2, size=1)[0]
        col = path[-1][1] + opt
        path.append((line, col))

    return path

def select_dynamic_programming_path(E):
    M=np.zeros(E.shape)
    M[0,:]=E[0,:]
    for i in range(1,M.shape[0]):
        for j in range(0,M.shape[1]):
            if j==0:
                M[i,j]=E[i,j]+min(M[i-1,j],M[i-1,j+1])
            elif j==E.shape[1]-1:
                M[i,j]=E[i,j]+min(M[i-1,j],M[i-1,j-1])
            else:
                M[i,j]=E[i,j]+min(M[i-1,j-1],M[i-1,j],M[i-1,j+1])
    line=M.shape[0]-1
    col=np.argmin(M[line,:])
    path=[0 for i in range(line+1)]
    path[line]=(line,col)
    for line in  range(M.shape[0]-2,-1,-1):
        if col==0:
            if M[line,0]>M[line,1]:
                new_col=1
            else:
                new_col=0
        elif col==E.shape[1]-1:
            if M[line,col]>M[line,col-1]:
                new_col=col-1
            else:
                new_col=col
        else:
            neighbours=np.array([M[line,col-1],M[line,col],M[line,col+1]])
            new_col=col+np.argmin(neighbours)-1
        path[line]=(line,new_col)
        col=new_col
    return path


def select_path(E, method):
    if method == 'aleator':
        return select_random_path(E)
    elif method == 'programareDinamica':
        return select_dynamic_programming_path(E)
    else:
        print('The selected method %s is invalid.' % method)
        sys.exit(-1)

#===run_project.py===

"""
    PROIECT
    REDIMENSIONEAZA IMAGINI.
    Implementarea a proiectului Redimensionare imagini
    dupa articolul "Seam Carving for Content-Aware Iamge Resizing", autori S. Avidan si A. Shamir
"""

from parameters import *
from resize_image import *
import matplotlib.pyplot as plt

image_name = '../data/lac.jpg'
params = Parameters(image_name)

# seteaza optiunea de redimenionare
# micsoreazaLatime, micsoreazaInaltime, amplificaContinut, eliminaObiect
params.resize_option = 'eliminaObiect'
# numarul de pixeli pe latime
params.num_pixels_width = 75
# numarul de pixeli pe inaltime
params.num_pixel_height = 50
# afiseaza drumul eliminat
params.show_path = True
# metoda pentru alegerea drumului
# aleator, greedy, programareDinamica
params.method_select_path = 'programareDinamica'
params.factor_amplification=1.20
resized_image = resize_image(params)
resized_image_opencv = cv.resize(params.image, (resized_image.shape[1], resized_image.shape[0]))

f, axs = plt.subplots(2, 2, figsize=(15, 15))
plt.subplot(1, 3, 1)
plt.imshow(np.uint8(params.image[:, :, [2, 1, 0]]))
plt.xlabel('original')

plt.subplot(1, 3, 2)
plt.imshow(np.uint8(resized_image_opencv[:, :, [2, 1, 0]]))
plt.xlabel('OpenCV')

plt.subplot(1, 3, 3)
plt.imshow(resized_image[:, :, [2, 1, 0]])
plt.xlabel('My result')
cv.imwrite('test.png', resized_image)
plt.show()

#===parameters.py===

import cv2 as cv
import sys
import numpy as np


class Parameters:

    def __init__(self, image_name):
        self.image_name = image_name
        self.image = cv.imread(image_name)
        if self.image is None:
            print('The image name %s is invalid.' % self.image_name)
            sys.exit(-1)
        self.image = np.float32(self.image)
        self.resize_option = 'micsoreazaLatime'
        self.num_pixels_width = 50
        self.num_pixel_height = 50
        self.show_path = False
        self.color_path = (0, 0, 255)
        self.method_select_path = 'greedy'
        self.factor_amplification = 1.05