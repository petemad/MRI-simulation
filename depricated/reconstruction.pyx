from rotation import rotateX, gradientXY, rotateZ
from RD import recovery, decay
import numpy as np
from se import dephasing


def GRE_reconstruct_image(phantomSize,vectors,TE,TR,FA,T2,T1):
    kSpace = np.zeros((phantomSize, phantomSize), dtype=np.complex_)
    for i in range(0, round(phantomSize)):
        rotatedMatrix = rotateX(vectors, FA)
        decayedRotatedMatrix = decay(rotatedMatrix, T2, TE)

        for j in range(0,phantomSize):
            stepX = (360 / phantomSize) * i
            stepY = (360 / phantomSize) * j
            phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
            sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
            sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
            valueToAdd = np.complex(sigmaX, sigmaY)
            kSpace[i, j] = valueToAdd

        decayedRotatedMatrix[:, :, 0] = 0
        decayedRotatedMatrix[:, :, 1] = 0
        print(i)
        vectors = recovery(decayedRotatedMatrix, T1, TR)

    kSpace = np.fft.fft2(kSpace)
    return kSpace


def spin_echo_reconstruct_image(phantomSize,vectors,TE,TR,FA,T2,T1):
            
    kSpace = np.zeros((phantomSize, phantomSize), dtype=np.complex_)
    
    vectors = rotateX(vectors, FA)
        # dephasing 
    dephase = dephasing(phantomSize,vectors)
        #dephase = decay(dephase,self.T2,self.TE)
        # rephasing
    vectors = rotateX(dephase, 2*FA) 

    for i in range(0, round(phantomSize)):
        for j in range(0, phantomSize):
            stepX = (360 / phantomSize) * i
            stepY = (360 / phantomSize) * j
            phaseEncodedMatrix = gradientXY(vectors, stepY, stepX)
            sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
            sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
            valueToAdd = np.complex(sigmaX, sigmaY)
            kSpace[i, j] = valueToAdd
        print(i)
    kSpace = np.fft.fft2(kSpace)
    return kSpace

def SSFP_reconstruct_image(phantomSize,vectors,TE,TR,FA,T2,T1):
    kSpace = np.zeros((phantomSize, phantomSize), dtype=np.complex_)
        
    vectors = rotateX(vectors, FA)
    rotatedMatrix = vectors
    for i in range(0, round(phantomSize)):

        decayedRotatedMatrix = decay(rotatedMatrix, T2, TE)

        for j in range(0, phantomSize):
            stepX = (360 / phantomSize) * i
            stepY = (360 / phantomSize) * j
            phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
            sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
            sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
            valueToAdd = np.complex(sigmaX, sigmaY)
            kSpace[i, j] = valueToAdd

        rotatedMatrix = recovery(decayedRotatedMatrix,T1,TE)
        print(i)
        if i % 2 == 0:
            vectors = rotateX(rotatedMatrix, -1 * FA * 2)
        else:
            vectors = rotateX(rotatedMatrix, FA * 2)

    kSpace = np.fft.fft2(kSpace)
    return kSpace
