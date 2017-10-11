import numpy as np
import cv2
from scipy import fftpack


img = cv2.imread("Lenna.png",0) #gray
width,height = img.shape
width = int (width/8)
height = int(height/8)

def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

def get_2d_idct(coefficients):
    """ Get 2D Inverse Cosine Transform of Image
    """
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')


def create_mask(porcentaje):
    num = porcentaje*64/100 #numero de unos en la matriz
    mask = np.ones((8,8))*0.1
    cont = 0
    for i in range(0,8):
        if cont == num:
            return mask[0:8,0:8]
        else:

            for j in range(0,i):
                mask[i, j] = 1
                cont = cont + 1
                if cont == num:
                    return mask[0:8, 0:8]

    return mask





eo = create_mask(50)
print(eo)
'''
mask= np.zeros((8,8))
mask[0,0]=1
img_reducida = np.zeros(img.shape,dtype='uint8')

for i in range(0,width):
    for j in range(0,height):
        block = img[8*i:8*(i+1),(j*8):8*(j+1)]
        block_dct = get_2D_dct(block)
        block_dct = np.multiply(mask,block_dct) #reduciendo
        img_reducida[8*i:8*(i+1),(j*8):8*(j+1)] = get_2d_idct(block_dct)

cv2.imshow("imagen original",img)
cv2.imshow("imagen reducida",img_reducida)
cv2.waitKey()


'''








