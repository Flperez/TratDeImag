import numpy as np
import cv2
import matplotlib.pyplot as plt



def calcula_SSD(xr,yr,xl,yl,f,h,block):
    sum = 0


    for i in range(-int(block[0]/2),+int(block[0]/2)+1):
        for j in range(-int(block[1]/2),+int(block[1]/2)+1):
            sum = sum + (f[xr+i,yr+j]-h[xl,yl])**2
    sum = 1+sum
    SSD = 1/sum
    '''
    optimizar con los np.sum 
    '''
    return SSD




block=(11,11)
max_dispar = 50

if __name__=="__main__":
    imgR = cv2.imread("b1.jpg",0)
    imgL = cv2.imread("b2.jpg",0)
    disparidad = np.zeros(imgR.shape,dtype='uint8')

    for x in range(int(block[0]/2),imgR.shape[0]-int(block[0]/2)):
        for y in range(int(block[1]/2),imgR.shape[1]-int(block[1]/2)):
            #Para el pixel "x" e "y"

            flag = False
            for it in range(0,max_dispar):
                if x-it<0: #nos hemos pasado de la imagen de la izquierda
                    break
                else:
                    SSD = calcula_SSD(xr=x-it,yr=y,
                                          xl=x,yl=y,
                                          f=imgR,h=imgL,
                                          block=block)
                    if flag == False:
                        min = SSD
                        flag = True
                    else:
                        if SSD<min:
                            min=SSD
                        else:
                            # la iteracion anterior es la que tiene el SSD menor

                            break

            disparidad[x, y] = it - 1



            print("Pos [", x, ",", y, "]"," disparidad: ",disparidad[x,y])


    max_dispar = np.amax(disparidad)
    disparidad = int(255/max_dispar) * disparidad

    cv2.imwrite("disparidad.png", disparidad)

    plt.subplot(131)
    plt.imshow(imgR,cmap='gray')
    plt.title("Right image")
    plt.subplot(132)
    plt.imshow(imgL,cmap='gray')
    plt.title("Left image")
    plt.subplot(133)
    plt.imshow(disparidad,cmap='gray')
    plt.title("Disparidad")

    plt.show()






