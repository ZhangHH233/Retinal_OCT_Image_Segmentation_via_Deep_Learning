from keras import backend as K
import numpy as np
import cv2 as cv

class FourierDescriptors:
    def __init__(self, N):
        self.N = N
        
    def calculate_descriptors(self, y_true):        
        descriptors = []
        y_c = np.uint8(y_true) # dimension of y_true is 2        
        
        contours, _ = cv.findContours(y_c.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        for contour in contours:
            descriptors.append(self.calculate_fourier_descriptors(contour))
        return descriptors, contours

    def calculate_fourier_descriptors(self, contour):
        return self.calculate_fourier_descriptors_center(contour)

    def calculate_fourier_descriptors_center(self, contour):
        center = self.calculate_center(contour)
        delta = []
        l = []
        for i in range(1, len(contour)+1):
            point1 = (contour[i-1][0][1], contour[i-1][0][0])
            point2 = (contour[i%len(contour)][0][1], contour[i%len(contour)][0][0])
            d1 = np.sqrt((point1[0]-center[0])**2+(point1[1]-center[1])**2)
            d2 = np.sqrt((point2[0]-center[0])**2+(point2[1]-center[1])**2)
            delta.append(d1-d2)
            d3 = np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
            l.append(d3)
        for i in range(1, len(l)):
            l[i] += l[i-1]
        A = []
        for i in range(1, self.N+1):
            A.append(self.calculate_fourier_coefficients(i, l, delta))
        return np.array(A)
    
    def calculate_center(self, contour):
        x, y, num = 0, 0, 0
        for pixel in contour:
            num += 1
            x += pixel[0][1]
            y += pixel[0][0]
        return x/num, y/num
    
    def calculate_fourier_coefficients(self, k, l, delta):
        a, b = 0, 0
        L = l[-1]
        for i in range(len(l)):
            if delta[i] != 0:
                a += delta[i]*np.sin((2*np.pi*k*l[i])/L)
                b += delta[i]*np.cos((2*np.pi*k*l[i])/L)
        a = a/(k*np.pi)
        b = -b/(k*np.pi)
        return np.sqrt(a*a+b*b)
        
        
N = 1
fd_obj = FourierDescriptors(N)

img = (cv.imread(".png")[:,:,0] > 0).astype(np.int_) # gold map
h, w = img.shape
maps = np.zeros((h,w,N))
shrinked_img = img.copy()

while(True):
    amp, cnt = fd_obj.calculate_descriptors(shrinked_img)
    if cnt==[]:
        break
    maps2 = np.zeros((h,w,N))
    for i in range(len(cnt)):
        if len(cnt[i]) != 0:
            amplitude = amp[i]
            for j in range(cnt[i].shape[0]):
                y = cnt[i][j,0,0]
                x = cnt[i][j,0,1]            
                maps2[x, y, :] = amplitude
                shrinked_img[x, y] = 0
    maps += maps2

for i in range(N):
    name = "fdmap" + str(i+1)
    np.savetxt(name, maps[:,:,i], fmt='%.8g')