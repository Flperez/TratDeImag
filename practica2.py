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
            image[:,j] = 255*(np.cos((2*mt.pi)*(j/imSize[0])*frec))

    if functType == "sin" :

        for j in range(0,imSize[1]):
            image[:,j] = 255*(np.sin((2*mt.pi)^2*(j/imSize[0])*frec))

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





    return imageFFT

def immagphase(freq):
    '''

    :param freq:
    :return:
    '''

    angle = np.angle(freq)
    mag = np.absolute(freq)
    mag = np.log(mag+1e-7)
    print("angle: ",angle)
    print("mag: ",mag)

    x = mag*np.cos(angle)
    y = mag*np.sin(angle)

    plt.plot(y,x,'x')
    plt.title('Representacion Re, Im')
    plt.xlabel('Re(log)')
    plt.ylabel('Im')
    plt.show()


    return

if __name__ == "__main__":

    image = imGenerator((100,100),"cos",1,0)
    freq = imfft(image)
    cv2.imshow("image", image)
    cv2.waitKey()

    immagphase(freq)







