import numpy as np
import cv2
from scipy import fftpack
import matplotlib.pyplot as plt




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

    img = cv2.imread("Lenna.png")  # gray

    img = cv2.cvtColor(src=img,code=cv2.COLOR_RGB2YCrCb)
    img = img[:,:,0]
    print(img.shape)
    size = img.shape
    width, height = img.shape

    width = int(width / 8)
    height = int(height / 8)
    size2 = 8*width,8*height

    if size != size2:
        img = cv2.resize(src=img,dsize=size)
        print("Redimensionado a ",size2)


    mask= np.zeros((8,8))
    mask[0,0]=1

    mask2=np.zeros((8,8))
    mask2[0:7,0:7]=np.rot90(np.tri(7, 7, -3, dtype=int).T)

    img_reducida = np.zeros(img.shape,dtype='uint8')
    img_reducida2 = np.zeros(img.shape,dtype='uint8')

    for i in range(0,width):
        for j in range(0,height):
            block = img[8*i:8*(i+1),(j*8):8*(j+1)]
            block_dct = get_2D_dct(block)
            block_dct = np.multiply(mask,block_dct) #reduciendo
            img_reducida[8*i:8*(i+1),(j*8):8*(j+1)] = get_2d_idct(block_dct)


            block_dct2 = get_2D_dct(block)
            block_dct2 = np.multiply(mask2, block_dct2)  # reduciendo
            img_reducida2[8 * i:8 * (i + 1), (j * 8):8 * (j + 1)] = get_2d_idct(block_dct2)

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('Imagen original')
    plt.subplot(132)
    plt.imshow(img_reducida, cmap='gray')
    plt.title('Imagen reducida 1/64')
    plt.subplot(133)
    plt.imshow(img_reducida2, cmap='gray')
    plt.title('Imagen reducida 10/64')
    plt.show()

