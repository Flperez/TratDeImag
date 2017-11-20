import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import jit

@jit
def calcula_SSD(xr,yr,xl,yl,f,h,block):

    inc = int(block[0] / 2)
    block_f = f[xr - inc:xr + inc+1, yr - inc:yr + inc+1]
    block_h = h[xl - inc:xl + inc+1, yl - inc:yl + inc+1]


    #SSD = 1/(1+np.sum((block_f-block_h)**2))
    SSD = np.sum((block_f-block_h)**2)


    return SSD





block=(11,11)
max_dispar = 50

if __name__=="__main__":
    imgR = cv2.imread("a2.jpg",0)
    imgL = cv2.imread("a1.jpg",0)
    disparidad = np.zeros(imgR.shape,dtype='uint8')

    for x in range(int(block[0]/2),imgR.shape[0]-int(block[0]/2)):
        for y in range(int(block[0]/2),imgR.shape[1]-int(block[0]/2)-1):
            #Para el pixel "x" e "y"

            flag = False
            for it in range(0,max_dispar):
                if y-it<=5: #nos hemos pasado de la imagen de la izquierda
                    break
                else:
                    SSD = calcula_SSD(xr=x,yr=y-it,
                                          xl=x,yl=y,
                                          f=imgR,h=imgL,
                                          block=block)

                    if flag == False:
                        min = SSD
                        flag = True
                    else:
                        if SSD < min:
                            min = SSD
                            disparidad[x, y] = it





            print("Pos [", x, ",", y, "]"," disparidad: ",disparidad[x,y],
                  " SSD: ",min)


    max_dispar = np.amax(disparidad)
    disparidad = int(255/max_dispar) * disparidad

    cv2.imwrite("disparidad.png", disparidad)
    plt.imshow(disparidad,cmap='gray')
    plt.title("Disparidad")

    plt.show()






