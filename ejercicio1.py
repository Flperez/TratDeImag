import numpy as np
import math as mt
import cv2
import matplotlib.pyplot as plt



def lowpassfilter(tam,frecCorte,n):
    '''

    :param tam:
    :param frecCorte:
    :param n:
    :return:
    '''
    H = np.zeros(tam)
    center = [0.5*i for i in tam]

    for u in range(0,tam[0]):
        for v in range(0,tam[1]):
            # distancia del centro a (u,v)
            D = mt.sqrt((u-center[0])**2+(v-center[1])**2)
            H[u,v]=1/((1+((D/frecCorte)**(2*n))))



    return H
def highpassfilter(tam,frecCorte,n):
    '''

    :param tam: size of image
    :param frecCorte:
    :param n:
    :return:
    '''

    H = 1 - lowpassfilter(tam=tam,frecCorte=frecCorte,n=n)

    return H


def imfft(image):
    '''

    :param image:
    :return:

    '''
    imageFFT = np.fft.fft2(image)
    imageFFT = np.fft.fftshift(imageFFT)
    return imageFFT

if __name__=="__main__":

    lenna = cv2.imread('Lenna.png', flags=0)
    lenna_fft = imfft(lenna)

    # Filtro de paso alto
    H_high = highpassfilter(tam=lenna_fft.shape, frecCorte=10, n=1)
    H_high_gray = np.multiply(255, H_high)
    H_high_gray = np.array(H_high_gray, dtype='uint8')

    result_fft_high = np.multiply(lenna_fft, H_high)
    result_high = np.array(abs(np.fft.ifft2(result_fft_high).real), dtype='uint8')

    plt.subplot(131)
    plt.imshow(lenna, cmap='gray')
    plt.title('Imagen')
    plt.subplot(132)
    plt.imshow(H_high_gray, cmap='gray')
    plt.title('Filtro de paso alto')
    plt.subplot(133)
    plt.imshow(result_high, cmap='gray')
    plt.title('Imagen*Filtro')
    plt.show()
