import numpy as np

def tagging(signal,phantomSize, step = 4):
        
    for k in range(0,phantomSize,step):
        for m in range(phantomSize):
            Gx= (m/phantomSize)*(2*np.pi) # rows           
            signal[k][m][2]=signal[k][m][2]*np.sin(Gx)
    
    return signal
