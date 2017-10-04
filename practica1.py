import numpy as np
import math as mt
import matplotlib.pyplot as plt

num_datos = 1000
t = np.linspace(0,2*mt.pi,num=num_datos)
f = np.zeros(len(t))
ts =  (2*mt.pi)/num_datos



for i in range (0,len(t)):
    f[i]=8+3*mt.cos(t[i])+2*mt.cos(2*t[i])+mt.cos(3*t[i])+2*mt.sin(t[i])+4*mt.sin(2*t[i])+3*mt.sin(3*t[i])

print("Metodo rectangulo")
a0=0
an=np.zeros(3)
bn=np.zeros(3)
for i in range(0,len(t)-1):
    a0=a0+(f[i])*ts

    for j in range(0,3):
        an[j]=an[j]+(f[i]*mt.cos((j+1)*t[i]))*ts
        bn[j]=bn[j]+(f[i]*mt.sin((j+1)*t[i]))*ts


a0=a0/(2*mt.pi)
for j in range(0, 3):
    an[j] = an[j] / mt.pi
    bn[j] = bn[j] / mt.pi


print("a0:",a0,"a: ",an,"bn: ",bn)


print("Metodo trapecio")
a0=0
an=np.zeros(3)
bn=np.zeros(3)
for i in range(0,len(t)-1):
    a0=a0+((f[i]+f[i+1])/2)*ts
    for j in range(0,3):
        an[j]=an[j]+((f[i]+f[i+1])/2)*mt.cos((j+1)*t[i])*ts
        bn[j]=bn[j]+((f[i]+f[i+1])/2)*mt.sin((j+1)*t[i])*ts


a0=a0/(2*mt.pi)
for j in range(0, 3):
    an[j] = an[j] / mt.pi
    bn[j] = bn[j] / mt.pi

plt.figure()
plt.plot(f)
plt.show()

print("a0:",a0,"a: ",an,"bn: ",bn)