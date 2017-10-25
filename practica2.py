import numpy as np
import math as mt
import cv2
import matplotlib.pyplot as plt

def imGenerator(imSize, functType,frec,orientation ):
    '''
    :param imSize:
    :param functType:
    :param frec: frecuencia de cambio
    :param orientation:
    :return: imagen
    '''
    image = np.zeros(imSize,dtype='uint8')

    if functType == "cos" :

        for j in range(0,imSize[1]):
            image[:,j] = 255*abs(np.cos((mt.pi)*(j/imSize[1])*frec))

    if functType == "sin" :

        for j in range(0,imSize[1]):
            image[:,j] = 255*abs(np.sin((mt.pi)^2*(j/imSize[1])*frec))

    if orientation == 0:
        return image
    else:
        return np.rot90(image)



def imfft(image):
    '''

    :param image:
    :return:

    '''
    imageFFT = np.fft.fft2(image)
    imageFFT = np.fft.fftshift(imageFFT)
    return imageFFT

def immagphase(freq):
    '''
    No se como es
    :param freq:
    :return:
    '''

    angle = np.angle(freq)
    mag = np.absolute(freq)
    #mag = np.log(mag+1e-7)


    x = mag*np.cos(angle)
    y = mag*np.sin(angle)

    plt.plot(y,x,'x')
    plt.title('Representacion Re, Im')
    plt.xlabel('Re(log)')
    plt.ylabel('Im')
    plt.show()


    return

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

if __name__ == "__main__":

    image = imGenerator((100,150),"cos",1,1)
    freq = imfft(image)
    image_freq = np.array(abs(255*(freq).real),dtype='uint8')


    lenna = cv2.imread('Lenna.png', flags=0)
    lenna_fft = imfft(lenna)

    # Filtro de paso bajo
    '''

    H_low = lowpassfilter(tam=lenna_fft.shape,frecCorte=10,n=1)
    H_low_gray = np.multiply(255,H_low)
    H_low_gray = np.array(H_low_gray,dtype='uint8')

    lenna_gray = np.multiply(255,lenna_fft)
    lenna_gray = np.array(lenna_gray.real, dtype='uint8')



    result_fft_low = np.multiply(lenna_fft,H_low)
    result_low = np.array(abs(np.fft.ifft2(result_fft_low).real),dtype='uint8')

    plt.subplot(131)
    plt.imshow(lenna, cmap='gray')
    plt.title('Imagen')
    plt.subplot(132)
    plt.imshow(H_low_gray, cmap='gray')
    plt.title('Filtro paso bajo')
    plt.subplot(133)
    plt.imshow(result_low, cmap='gray')
    plt.title('Imagen*filtro')
    plt.show()
    '''

    # Filtro de paso alto
    H_high = highpassfilter(tam=lenna_fft.shape,frecCorte=10,n=1)
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
    plt.imshow(result_high,cmap='gray')
    plt.title('Imagen*Filtro')
    plt.show()




    # -----------------Sobel--------------------------- #

    ## Dominio de la frecuencia
    # Sobel en eje x
    mask_x = sobelmask(tam=lenna_fft.shape, orientation='x')
    result_fft_x = np.multiply(lenna_fft,mask_x)
    result_x= np.array(abs(np.fft.ifft2(result_fft_x).real),dtype='uint8')

    # Sobel en eje y
    mask_y = sobelmask(tam=lenna_fft.shape, orientation='y')
    result_fft_y = np.multiply(lenna_fft, mask_y)
    result_y = np.array(abs(np.fft.ifft2(result_fft_y).real), dtype='uint8')




    ## Dominio de la imagen
    imgSobelx = sobel_dom_imagen(img=lenna,orientation='x')
    imgSobely = sobel_dom_imagen(img=lenna,orientation='y')

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