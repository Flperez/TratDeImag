import numpy as np
import math as mt
import cv2
import matplotlib.pyplot as plt

def imfft(image):
    '''

    :param image:
    :return:

    '''
    imageFFT = np.fft.fft2(image)
    imageFFT = np.fft.fftshift(imageFFT)
    return imageFFT

def sobelmask(tam,orientation):
    '''

    :param tam: size of image
    :param orientation: orientation of sobel mask
    :return: sobel FFT mask
    '''
    mask = np.zeros(tam)
    sobel = np.matrix('-1,0,1;-2,0,2;-1,0,1')

    if orientation != 'x':  #orientation = 'y'
        sobel = sobel.getT()

    mask[:sobel.shape[0], :sobel.shape[1]] = sobel  # zero padding
    mask = imfft(mask)

    return mask

def sobel_dom_imagen(img,orientation):
    sobel = np.matrix('-1,0,1;-2,0,2;-1,0,1')
    result= np.zeros(img.shape,dtype='uint8')

    if orientation != 'x':  # orientation = 'y'
        sobel = sobel.getT()

    for i in range(1,img.shape[0]-1):
        for j in range(1,img.shape[1]-1):

            block_img = img[i-1:i+2,j-1:j+2]
            result[i,j]=abs(0.25*np.sum(np.multiply(sobel,block_img)))

    return result


if __name__=="__main__":

    lenna = cv2.imread('Lenna.png', flags=0)
    lenna_fft = imfft(lenna)

    ## Dominio de la frecuencia
    # Sobel en eje x
    mask_x = sobelmask(tam=lenna_fft.shape, orientation='x')
    result_fft_x = np.multiply(lenna_fft, mask_x)
    result_x = np.array(abs(np.fft.ifft2(result_fft_x).real), dtype='uint8')

    # Sobel en eje y
    mask_y = sobelmask(tam=lenna_fft.shape, orientation='y')
    result_fft_y = np.multiply(lenna_fft, mask_y)
    result_y = np.array(abs(np.fft.ifft2(result_fft_y).real), dtype='uint8')

    ## Dominio de la imagen
    imgSobelx = sobel_dom_imagen(img=lenna, orientation='x')
    imgSobely = sobel_dom_imagen(img=lenna, orientation='y')

    _, axarr = plt.subplots(2, 2)
    axarr[0, 0].imshow(result_x, cmap='gray')
    axarr[0, 0].set_title('Sobel_x(Frecuencia)')
    axarr[0, 1].imshow(imgSobelx, cmap='gray')
    axarr[0, 1].set_title('Sobel_x(Imagen)')
    axarr[1, 0].imshow(result_y, cmap='gray')
    axarr[1, 0].set_title('Sobel_y(Frecuencia)')
    axarr[1, 1].imshow(imgSobely, cmap='gray')
    axarr[1, 1].set_title('Sobel_y(Imagen)')
    plt.show()