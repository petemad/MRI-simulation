import numpy as np
from math import cos, sin, pi
import qimage2ndarray
from PyQt5.QtGui import QPixmap


def gradientXY(matrix, stepY, stepX):
    shape = np.shape(matrix)
    rows = shape[0]
    cols = shape[1]
    newMatrix = np.zeros(shape)
    for i in range(0, rows):
        for j in range(0, cols):
            angle = stepY * j + stepX * i
            angle = (angle) * (pi / 180)
            newMatrix[i, j] = np.dot(
                np.array([[cos(angle), -1 * sin(angle), 0], [sin(angle), cos(angle), 0], [0, 0, 1]]), matrix[i, j])
    return newMatrix

def dephasing(size,vectors):
    for i in range(0, round(size)):
        for j in range(0, size):
            stepX = (360 / size) * i
            stepY = (360 / size) * j
            dephase = gradientXY(vectors, stepY, stepX)
    return dephase

def showKSpace(img):
    img = img[:]
    img = 20 * np.log(np.abs(img))

    qimg = qimage2ndarray.array2qimage(np.abs(img))
    self.ui.kspaceLbl.setPixmap(QPixmap(qimg))

def rephasing(size,vectors):
    for i in range(0, round(size)):
        for j in range(0, size):
            stepX = (360 / size) * i
            stepY = (360 / size) * j
            phaseEncodedMatrix = gradientXY(vectors, stepY, stepX)
            sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
            sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
            valueToAdd = np.complex(sigmaX, sigmaY)
            kSpace[i, j] = valueToAdd

        showKSpace(kSpace)
        print(i)
    return kSpace

def MYspin_echo_reconstruct_image(self):
            
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

    
        vectors = self.preparation(vectors)
        vectors = self.startup(vectors)
        
        vectors = rotateX(vectors, self.FA)

        # dephasing 
        dephase = dephasing(self.phantomSize,vectors)

        # rephasing
        vectors = rotateZ(dephase, 2*self.FA)
        for i in range(0, round(self.phantomSize)):
            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                phaseEncodedMatrix = gradientXY(vectors, stepY, stepX)
                sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd

            self.showKSpace(kSpace)
            print(i)

        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)
