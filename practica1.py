import numpy as np
import math as mt
import matplotlib.pyplot as plt


def calcula_area(f,ts,method='rectangulo'):
    area = 0
    if method=='rectangulo':
        for i in range(0, len(f) - 1):
            area = area + (f[i]) * ts
    else:    #trapecio
        for i in range(0, len(t) - 1):
            area = area + ((f[i] + f[i + 1]) / 2) * ts
    return area

def calcula_parametros(f,ts,num):

    an = np.zeros(num+1)
    bn = np.zeros(num)

    an[0] = calcula_area(f=f,ts=ts,method='rectangulo')
    an[0] =  (an[0]/(2*mt.pi))*mt.pi #lo ponemos en forma de array



    for i in range(0,len(f)-1):
        for j in range(0,num):
            an[j+1]=an[j+1]+(f[i]*mt.cos((j+1)*t[i]))*ts
            bn[j]=bn[j]+(f[i]*mt.sin((j+1)*t[i]))*ts

    an = np.multiply((1/ mt.pi),an)
    bn = np.multiply((1 / mt.pi), bn)
    return an,bn

if __name__ == "__main__":

    num_datos = 1000
    t = np.linspace(0,2*mt.pi,num=num_datos)
    f = np.zeros(len(t))
    g = np.zeros(len(t))
    ts =  (2*mt.pi)/num_datos



    for i in range (0,len(t)):
        f[i]=8+3*mt.cos(t[i])+2*mt.cos(2*t[i])+mt.cos(3*t[i])+2*mt.sin(t[i])+4*mt.sin(2*t[i])+3*mt.sin(3*t[i])


    an,bn = calcula_parametros(f=f,ts=ts,num=4)
    print(an,bn)

    t0=t
    t = np.linspace(mt.pi,  3*mt.pi, num=num_datos)
    for i in range (0,len(t)):
        g[i]=an[0]+an[1]*mt.cos(t[i])+an[2]*mt.cos(2*t[i])+\
             an[3]*mt.cos(3*t[i])+an[4]*mt.cos(4*t[i])+\
             +bn[0]*mt.sin(t[i])+bn[1]*mt.sin(2*t[i])+bn[2]*mt.sin(3*t[i])+\
             bn[3]*mt.sin(4*t[i])


    plt.figure()
    plt.plot(t0,f, label= 'fcn original')
    plt.plot(t,g,'--',label='fcn recreada')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=2, mode="expand", borderaxespad=0.)
    plt.grid(True)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.show()



