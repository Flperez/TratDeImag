import numpy as np
import cv2
from scipy import fftpack




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




if __name__=="__main__":

    img = cv2.imread("Lenna.png", 0)  # gray
    width, height = img.shape
    width = int(width / 8)
    height = int(height / 8)

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


    ############### Reduccion JPG ########################
    luminancia = np.array([[16,11,10,16,24,40,51,61],
                          [12,12,14,19,26,58,60,55],
                          [14,13,16,24,40,57,69,56],
                          [14,17,22,29,51,87,80,62],
                          [18,22,37,56,68,109,103,77],
                          [24,36,55,64,81,104,113,92],
                          [49,64,78,87,103,121,210,101],
                          [72,92,95,98,112,100,103,99]])
    imgJPG=img
    imgJPG=imgJPG-255*np.ones(imgJPG.shape)
    for i in range(0,width):
        for j in range(0,height):
            block = imgJPG[8*i:8*(i+1),(j*8):8*(j+1)]

            block_dct = get_2D_dct(block)
            #se divide por el factor de calidad
            block_dct=np.divide(block_dct,luminancia)
            #se redondea
            block_dct=np.array(block_dct,dtype='int8')

            block_dct = block_dct+255*np.ones(block_dct.shape)
            img_reducida[8 * i:8 * (i + 1), (j * 8):8 * (j + 1)] = get_2d_idct(block_dct)

    cv2.imshow("imagen original", img)
    cv2.imshow("imagen 'jpg'", img_reducida)
    cv2.waitKey()










