import numpy as np
import matplotlib.pyplot as plt

def punto1():
    fig_a, ax_a = plt.subplots(1,2,figsize=(10,5))
    ax_a[0].set_title("Rhodium Xray data")
    ax_a[0].set_xlabel("Wavelength (pm)")
    ax_a[0].set_ylabel("Intensity (mJy)")
    ax_a[1].set_title("Rhodium Xray data filtered")
    ax_a[1].set_xlabel("Wavelength (pm)")
    ax_a[1].set_ylabel("Intensity (mJy)")
    def cargaDatos():
        x,y = np.genfromtxt('Rhodium.csv',dtype = float,delimiter=',',skip_header=1,unpack=True)
        promx = 0
        for i in range(1,len(x)):
            promx+=x[i]-x[i-1]
        promx = promx/(len(x)-1)
        ax_a[0].scatter(x,y,s=5)
        return x,y,promx
    def limpiezaDatos():
        j = 0
        x,y,promx = cargaDatos()
        x1 = x[:3]
        y1 = y[:3]
        totalx = len(x)
        for i in range(2,totalx-2):
            dx1 = x[i-1] - x[i-2]
            dy1 = y[i-1] - y[i-2]
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            dx2 = x[i+2] - x[i+1]
            dy2 = y[i+2] - y[i+1]
            if (abs(dy1/dx1) + abs(dy2/dx2))*dx/2 >= abs(dy/dx)*promx/7:
                x1 = np.append(x1,[x[i]])
                y1 = np.append(y1,[[y[i]]])
            else:
                j+=1
        ax_a[1].scatter(x1,y1,s=5)
        print(f'1.a) Número de datos eliminados: {j}')
        fig_a.savefig("limpieza.pdf", format="pdf", bbox_inches="tight")
        return x1,y1
    def dosPicos():
        x1,y1 = limpiezaDatos()
        imax = np.argmax(y1)
        i = imax
        while(y1[i+1]-y1[i]<0):
            i+=1
        imax2 = np.argmax(y1[i:]) + i
        avg = 0
        for j in range(imax,imax2):
            dy = y1[j+1] - y1[j-1]
            dx = x1[j+1] - x1[j-1]
            avg += abs(dy / dx) if dx !=0 else 0
        avg /= imax2-imax
        right = True
        left = True
        r = imax2
        l = imax
        while(right or left):
            if(right):
                r+=1
                dy = y1[r] - y1[r - 1]
                dx = x1[r] - x1[r - 1]
                right = True if abs(dy/dx)>=avg/15 else False
            if(left):
                l-=1
                dy = y1[l+1] - y1[l]
                dx = x1[l+1] - x1[l]
                left = True if abs(dy / dx) >= avg/25 else False
        fig_b, ax_b = plt.subplots()
        ax_b.set_title("Rhodium Xray peaks")
        ax_b.set_xlabel("Wavelength (pm)")
        ax_b.set_ylabel("Intensity (mJy)")
        ax_b.scatter(x1[l:r+1],y1[l:r+1],s=5)
        fig_b.savefig("picos.pdf", format="pdf", bbox_inches="tight")
        print('1.b) Método: Tolerancia valor de las derivadas, partiendo de los picos')
        return i,imax,imax2,r,l,x1,y1
    def localizacion():
        i,imax,imax2,r,l,x1,y1 = dosPicos()
        ###pico1:
        h = (y1[imax]+y1[i])/2
        for j in range(l,imax+1):
            w1 = j if j==l or abs(y1[w1] - h)> abs(y1[j]-h) else w1
        for k in range(imax,i+1):
            w2 = k if k==imax or abs(y1[w2] - h)> abs(y1[k]-h) else w2
        wpico1 = x1[w2]-x1[w1]
        ###pico2:
        h = (y1[imax2] + y1[i]) / 2
        for j in range(i, imax2 + 1):
            w1 = j if j == i or abs(y1[w1] - h) > abs(y1[j] - h) else w1
        for k in range(imax2, r + 1):
            w2 = k if k == imax2 or abs(y1[w2] - h) > abs(y1[k] - h) else w2
        wpico2 = x1[w2] - x1[w1]

        print(wpico1,y1[imax])
        print(wpico2, y1[imax2])



    localizacion()




punto1()

