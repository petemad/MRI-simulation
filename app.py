####import pyximport; pyximport.install() not effective xD
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMessageBox
from mriui import Ui_MainWindow
from phantom import phantom
import numpy as np
import qimage2ndarray
import sys                     
import math
import threading
from multiprocessing.pool import ThreadPool
from rotation import rotateX, gradientXY, rotateZ
from RD import recovery, decay
from se import dephasing
from reconstruction import GRE_reconstruct_image , spin_echo_reconstruct_image , SSFP_reconstruct_image
import pyqtgraph as pg
from PyQt5.QtWidgets import QFileDialog
from math import sin, cos, pi
import csv
import sk_dsp_comm.sigsys as ss 
from sk_dsp_comm.sigsys import NRZ_bits2, m_seq
from copy import deepcopy

MAX_CONTRAST = 2
MIN_CONTRAST = 0.1
MAX_BRIGHTNESS = 100
MIN_BRIGHTNESS = -100
SAFETY_MARGIN = 10
MAX_PIXELS_CLICKED = 3

class ApplicationWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(ApplicationWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


        # Actions
        self.ui.comboSheppSize.currentTextChanged.connect(self.showPhantom)
        self.ui.comboViewMode.currentTextChanged.connect(self.changePhantomMode)
        self.ui.startSeq.clicked.connect(self.runSequence)
        self.ui.clearbtn.clicked.connect(self.clear_graphicalRep)
        self.ui.graphicalrepbtn.clicked.connect(self.showGraphicalRep)
        self.ui.artifact1btn.clicked.connect(self.art1btn_clicked)
        self.ui.artifact2btn.clicked.connect(self.art2btn_clicked)
        self.ui.FlipAngle.textChanged.connect(self.setFA)
        self.ui.startupText.textChanged.connect(self.setStartup)
        self.ui.prepText.textChanged.connect(self.setPrep)
        self.ui.TimeEcho.textChanged.connect(self.setTE)
        self.ui.TimeRepeat.textChanged.connect(self.setTR)
        self.ui.btnBrowse.clicked.connect(self.browse)
        self.ui.T2prepbtn.toggled.connect(self.preparationInintT2)
        self.ui.taggingbtn.toggled.connect(self.preparationInintTag)
        self.ui.irbtn.toggled.connect(self.preparationInintIR)
        self.ui.grebtn.toggled.connect(self.aquireGRE)
        self.ui.sebtn.toggled.connect(self.aquireSE)
        self.ui.ssfpbtn.toggled.connect(self.aquireSSFP)
        
        self.ui.UP.clicked.connect(self.up)
        self.ui.Down.clicked.connect(self.down)
        self.ui.left.clicked.connect(self.left)
        self.ui.Right.clicked.connect(self.right)
        self.ui.ZoomIn.clicked.connect(self.zoomIn)
        self.ui.ZoomOut.clicked.connect(self.zoomOut)
        self.ui.link.clicked.connect(self.link)


        # Mouse Events
        self.ui.phantomlbl.setMouseTracking(False)
        self.ui.phantomlbl.mouseMoveEvent = self.editContrastAndBrightness
        self.ui.phantomlbl.mouseDoubleClickEvent = self.pixelClicked

        # Scaling

        self.ui.phantomlbl.setScaledContents(True)
        self.ui.kspaceLbl.setScaledContents(True)

        # Plots
        self.ui.graphicsPlotT1.setMouseEnabled(False, False)
        self.ui.graphicsPlotT2.setMouseEnabled(False, False)

        self.ui.graphicsView_2.setMouseEnabled(False, False)
        self.pen = [QtGui.QPen(QtCore.Qt.green),QtGui.QPen(QtCore.Qt.red),QtGui.QPen(QtCore.Qt.yellow),QtGui.QPen(QtCore.Qt.blue)]
        self.Pen1 = [pg.mkPen('g'), pg.mkPen('r'), pg.mkPen('y'), pg.mkPen('b')]

        # initialization
        self.qimg = None
        self.img = None
        self.originalPhantom = None
        self.PD = None
        self.T1 = None
        self.T2 = None
        self.phantomSize = 512
        self.prep = None
        self.prepText = None
        self.acquisition = None 
        self.cycles = None

        self.FA = 90
        self.cosFA = 0
        self.sinFA = 1
        self.TE = 0.001
        self.TR = 0.5
        self.x = 0
        self.y = 0

        self.zoom=0
        self.drag_x=0
        self.drag_y=0
        self.zoomtype = 'PD'
        self.KsapceImage=None


        self.pixelsClicked = [(0, 0), (0, 0), (0, 0)]
        self.pixelSelector = 0

        # For Mouse moving, changing Brightness and Contrast
        self.lastY = None
        self.lastX = None

        # For Contrast Control
        self.contrast = 1.0
        self.brightness = 0

    def aquireSSFP(self):
        self.acquisition = 'ssfp'
    def aquireGRE(self):
            self.acquisition = 'gre'
    def aquireSE(self): 
            self.acquisition = 'se'

    def preparationInintT2(self):
        self.prep = 'T2prep'
        self.ui.preplbl.setText("t_wait")
        self.prepText = int(self.ui.prepText.toPlainText())
    def preparationInintTag(self):
        self.prep = 'tagging'
        self.ui.preplbl.setText("step")
        self.prepText = int(self.ui.prepText.toPlainText())
    def preparationInintIR(self):
        self.prep = 'ir'
        self.ui.preplbl.setText("T1_Null")
        self.prepText = float(self.ui.prepText.toPlainText())

    def browse(self):
        # Open Browse Window & Check
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open CSV", (QtCore.QDir.homePath()), "CSV (*.csv)")
        if fileName:
            # Check extension
            try:

                mat = np.genfromtxt(fileName, delimiter=',')
                row = math.floor(np.shape(mat)[0] / 3)
                col = np.shape(mat)[1]
                self.img = np.zeros([row, col])
                self.T1 = np.zeros([row, col])
                self.T2 = np.zeros([row, col])
                self.img = mat[0:row]
                self.PD = self.img
                self.originalPhantom = self.img
                self.phantomSize = row
                print(np.shape(self.img))
                self.T1 = mat[row:2 * row]
                print(np.shape(self.T1))
                self.T2 = mat[2 * row:3 * row + 1]
                print(np.shape(self.T2))
                self.showPhantomImage()
            except (IOError, SyntaxError):
                self.error('Check File Extension')

    def showPhantom(self, value):
        size = int(value)
        self.phantomSize = size
        self.ui.phantomlbl.phantomSize = size
        img = phantom(size)
        img = img * 255
        self.img = img
        self.PD = img
        self.T1 = phantom(size, 'T1')
        self.T2 = phantom(size, 'T2')
        self.originalPhantom = img
        self.pixelsClicked = [(0, 0), [0, 0], [0, 0]]
        self.showPhantomImage()

    def showPhantomImage(self):
        self.qimg = qimage2ndarray.array2qimage(self.img)
        self.ui.phantomlbl.setPixmap(QPixmap(self.qimg))

    def changePhantomMode(self, value):

        if value == "PD":
            self.img = self.PD
        if value == "T1":
            self.img = self.T1
        if value == "T2":
            self.img = self.T2

        self.img = self.img * (255 / np.max(self.img))
        self.originalPhantom = self.img
        self.showPhantomImage()

    def editContrastAndBrightness(self, event):
        if self.lastX is None:
            self.lastX = event.pos().x()
        if self.lastY is None:
            self.lastY = event.pos().y()
            return

        currentPositionX = event.pos().x()
        if currentPositionX - SAFETY_MARGIN > self.lastX:
            self.contrast += 0.01
        elif currentPositionX < self.lastX - SAFETY_MARGIN:
            self.contrast -= 0.01

        currentPositionY = event.pos().y()
        if currentPositionY - SAFETY_MARGIN > self.lastY:
            self.brightness += 1
        elif currentPositionY < self.lastY - SAFETY_MARGIN:
            self.brightness -= 1
        # Sanity Check
        if self.contrast > MAX_CONTRAST:
            self.contrast = MAX_CONTRAST
        elif self.contrast < MIN_CONTRAST:
            self.contrast = MIN_CONTRAST
        if self.brightness > MAX_BRIGHTNESS:
            self.brightness = MAX_BRIGHTNESS
        elif self.brightness < MIN_BRIGHTNESS:
            self.brightness = MIN_BRIGHTNESS

        self.img = 128 + self.contrast * (self.originalPhantom - 128)
        self.img = np.clip(self.img, 0, 255)

        self.img = self.img + self.brightness
        self.img = np.clip(self.img, 0, 255)
        self.showPhantomImage()

        self.lastY = currentPositionY
        self.lastX = currentPositionX

    def pixelClicked(self, event):
        if self.img is None:
            self.error('Choose Phantom First')
        else:
            self.pixelSelector = self.pixelSelector + 1
            self.pixelSelector = self.pixelSelector % 3
            t1Matrix = self.T1
            t2Matrix = self.T2
            self.x = event.pos().x()
            self.y = event.pos().y()
            self.ui.phantomlbl.pixelSelector += 1
            self.ui.phantomlbl.pixelSelector = self.ui.phantomlbl.pixelSelector % 3

            xt = self.ui.phantomlbl.frameGeometry().width()
            yt = self.ui.phantomlbl.frameGeometry().height()
            x = event.pos().x() * (self.phantomSize / xt)
            y = event.pos().y() * (self.phantomSize / yt)
            x = math.floor(x)
            y = math.floor(y)
            self.ui.t1lbl.setText(str(self.T1[x][y]))
            self.pixelsClicked.append((x, y))
            self.ui.phantomlbl.pixelsClicked.append((x, y))
            if len(self.pixelsClicked) > MAX_PIXELS_CLICKED:
                self.pixelsClicked.pop(0)
            if len(self.ui.phantomlbl.pixelsClicked) > MAX_PIXELS_CLICKED:
                self.ui.phantomlbl.pixelsClicked.pop(0)
            self.update()
            # self.paintEvent(event)
            t1graph = self.ui.graphicsPlotT1
            t2gragh = self.ui.graphicsPlotT2
            ernstGgraph = self.ui.graphicsView_2
            t1graph.clear()
            t2gragh.clear()
            ernstGgraph.clear()


            for pixelSet in self.pixelsClicked:
                x = pixelSet[0]
                y = pixelSet[1]
                if self.pixelSelector == 0:
                    color = 'r'
                if self.pixelSelector == 1:
                    color = 'b'
                if self.pixelSelector == 2:
                    color = 'y'
                t1 = t1Matrix[y][x]
                t2 = t2Matrix[y][x]
                self.plotting(color, t1 * 1000, t2 * 1000)
                self.pixelSelector += 1
                self.pixelSelector = self.pixelSelector % 3
                threading.Thread(target=self.ernst, args=(color, t1*1000, t2*1000)).start()
                         
#### Plotting #######################################################################
    def plotting(self, color, T1=1000, T2=45):
        t1graph = self.ui.graphicsPlotT1
        t2gragh = self.ui.graphicsPlotT2
        # theta = self.FA * pi / 180
        t = np.linspace(0, 10000, 1000)
        t1graph.plot(np.exp(-t / T1) * self.cosFA + 1 - np.exp(-t / T1), pen=pg.mkPen(color))
        t2gragh.plot(self.sinFA * np.exp(-t / T2), pen=pg.mkPen(color))

    def ernst(self,color, T1=1000, T2=45 ):
        ert = self.ui.graphicsView_2  
        if self.acquisition == 'gre' or self.acquisition == 'se':   
            angle = np.arange(0,180,1)
            anrad = angle /180 * np.pi
            ert.plot(np.sin(anrad)*(1-np.exp(-self.TR*1000/T1))*np.exp(-self.TE*1000/T2)/(1-(np.cos(anrad)*np.exp(-self.TR*1000/T1))), pen=pg.mkPen(color))
        if self.acquisition == 'ssfp':
            kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
            vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
            vectors[:, :, 2] = 1
            
            angle = np.arange(0,180,5)
            output = np.zeros(36)
            counter = -1
            for fa in range(0,180,5):
                print(fa)
                counter += 1
                rotatedMatrix = rotateX(vectors, fa)
                decayedRotatedMatrix = decay(rotatedMatrix,self.T2,self.TE)
                
                for j in range(0, self.phantomSize):
                    stepX = (360 / self.phantomSize) * 20
                    stepY = (360 / self.phantomSize) * j
                    phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
                    sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                    sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                    valueToAdd = np.complex(sigmaX, sigmaY)
                    kSpace[0, j] = valueToAdd
                rotatedMatrix = recovery(rotatedMatrix,self.T1,self.TR)
                output[counter] = np.average(np.abs(kSpace))
            ert.plot(angle,output,pen=pg.mkPen(color))
            


########### PREP #################################################################

    def preparation(self, signal):
        if self.prep == 'T2prep':
            output = self.T2prep(signal,self.prepText)
        elif self.prep == 'tagging':
            output = self.tagging2(signal,int(self.prepText))
        elif self.prep == 'ir':
            output = self.IR(signal, self.prepText)
        else :
            return signal
        return output

    def tagging(self,signal,step = 4):
        
        for k in range(0,self.phantomSize,step):
            for m in range(self.phantomSize):
                Gx= (m/self.phantomSize)*((pi-1)/2) # rows   # reduce to less than 90 is better        
                signal[k][m][2]=signal[k][m][2]*np.sin(Gx)
        signal[:,:,0] = 0
        signal[:,:,1] = 0
        return signal
    
    def tagging2(self,signal,step = 4): #te=0.008  tr=0.1 freq=60 ssfp
        
        signal = rotateX(signal,90)                 
        signal = gradientXY(signal,0,180/self.phantomSize*step)
        signal = rotateX(signal,-90)               
        signal[:,:,0] = 0
        signal[:,:,1] = 0
        return signal

    def IR(self,signal,T1cancel = 1000):
        
        TE = T1cancel * np.log(2)
#        self.TE = T1cancel * np.log(2) 
        signal = rotateX(signal , 180)
        signal = recovery(signal,self.T1,TE)
                 
        return signal  
         
    def T2prep(self, signal, t_wait) :

        signal = rotateX(signal , 90)
        signal = decay(signal,self.T2,t_wait)
        signal = rotateX(signal , -90)

        return signal 
####################################################################################### END PREP ###############

    def setFA(self, value):
        print(value)
        try:
            value = int(value)
            self.FA = value
            self.cosFA = cos(self.FA * pi / 180)
            self.sinFA = sin(self.FA * pi / 180)
        except:
            self.error("FA must be a number")

    def setTE(self, value):
        print(value)
        try:
            value = float(value)
            self.TE = value
        except:
            self.error("TE must be a float")

    def setTR(self, value):
        print(value)
        try:
            value = float(value)
            self.TR = value
        except:
            self.error("TR must be a float")

    def setStartup(self):
        try:
            self.cycles = int(self.ui.startupText.toPlainText())
        except :
            self.error("Number of preparation cycles must be an integer")


    def setPrep(self):
        try :
            self.prepText = float(self.ui.prepText.toPlainText())
        except ValueError : 
            self.prepText = int(self.ui.prepText.toPlainText())
        except :
            self.error("Insert preparation value")

    def startup(self,signal):
        try :
            for i in range(self.cycles):
                signal = rotateX(signal, self.FA)
                signal = decay(signal, self.T2, self.TE)
                signal = recovery(signal, self.T1, self.TR)
        except:
            return signal    
        return signal 

    def art1btn_clicked (self) :
        self.acquisition = 'artifact1'
        self.runSequence()
    def art2btn_clicked(self) :
        self.acquisition = 'artifact2'
        self.runSequence()

    def runSequence(self):
        if self.img is None:
            self.error('Choose a phantom first')
            return 
        if self.acquisition is None:
            self.error('Choose acquisition sequence frist')
            return 
        else:
            if self.acquisition == 'se':
                threading.Thread(target=self.spin_echo_reconstruct_image).start()
            elif self.acquisition == 'ssfp': ## worked using thread and cython ##time = 3m15.804s
                #v = self.prep_startup()  this took 
                #pool = ThreadPool(processes=1)
                #asyncResult = pool.apply_async(SSFP_reconstruct_image, (self.phantomSize,v,self.TE,self.TR,self.FA,self.T2,self.T1))
                #kspc = asyncResult.get()
                #self.showReconstructedImage(kspc)
                threading.Thread(target=self.SSFP_reconstruct_image).start() # ##time= 2m42.520s
            elif self.acquisition == 'gre':
                threading.Thread(target=self.GRE_reconstruct_image).start()
            elif self.acquisition == 'artifact1' :
                threading.Thread(target=self.artifact1).start()
            elif self.acquisition == 'artifact2' :
                threading.Thread(target=self.artifact2).start()
            return

    def prep_startup(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        vectors = self.startup(vectors)
        vectors = self.preparation(vectors)

        return vectors

    def GRE_reconstruct_image(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

     ###### Prep ##########################################################################################
        vectors = self.startup(vectors)
        vectors = self.preparation(vectors)

        for i in range(0, round(self.phantomSize)):
            rotatedMatrix = rotateX(vectors, self.FA)
            decayedRotatedMatrix = decay(rotatedMatrix, self.T2, self.TE)

            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
                sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd

            decayedRotatedMatrix[:, :, 0] = 0
            decayedRotatedMatrix[:, :, 1] = 0

            self.showKSpace(kSpace)
            print(i)
            vectors = recovery(decayedRotatedMatrix, self.T1, self.TR)

        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)

    
    def spin_echo_reconstruct_image(self):
            
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        vectors = self.preparation(vectors)
        vectors = self.startup(vectors)
        
        vectors = rotateX(vectors, self.FA)
        # dephasing 
        dephase = dephasing(self.phantomSize,vectors)
        #dephase = decay(dephase,self.T2,self.TE)
        # rephasing
        vectors = rotateX(dephase, 2*self.FA) 

        for i in range(0, round(self.phantomSize)):
            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                phaseEncodedMatrix = gradientXY(vectors, stepY, stepX)
                sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd
            #vectors =recovery(vectors,self.T1,self.TR)

            self.showKSpace(kSpace)
            print(i)

        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)

    def SSFP_reconstruct_image(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        
########## PREP ##########################################################33
        vectors = self.startup(vectors)
        vectors = self.preparation(vectors)

        vectors = rotateX(vectors, self.FA)
        rotatedMatrix = vectors
        for i in range(0, round(self.phantomSize)):

            decayedRotatedMatrix = decay(rotatedMatrix, self.T2, self.TE)

            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
                sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd

            rotatedMatrix = recovery(decayedRotatedMatrix,self.T1,self.TE)

            self.showKSpace(kSpace)
            print(i)
            if i % 2 == 0:
                vectors = rotateX(rotatedMatrix, -1 * self.FA * 2)
            else:
                vectors = rotateX(rotatedMatrix, self.FA * 2)

        #kSpace = np.fft.fftshift(kSpace)
        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)


########## Artifacts #########################################################################
    def artifact1 (self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        vectors = rotateX(vectors, 60)
        vectors = recovery(vectors, self.T1, 1000)

        for i in range(0, round(self.phantomSize)):
            rotatedMatrix = vectors
            decayedRotatedMatrix = decay(rotatedMatrix, self.T2, 45)

            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
                sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd

            self.showKSpace(kSpace)
            print(i)
            if i % 2 == 0:
                vectors = rotateX(rotatedMatrix, -1 * 60 * 2)
            else:
                vectors = rotateX(rotatedMatrix, 60 * 2)

        kSpace = np.fft.fftshift(kSpace)
        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)

        #kSpace = np.fft.fftshift(kSpace)
        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)

    def artifact12 (self): ## (;:;:;:)
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1    
   ####### PREP ##########################################################33
        vectors = self.startup(vectors)
        vectors = self.preparation(vectors)

        vectors = rotateX(vectors, 20)
        rotatedMatrix = vectors
        for i in range(0, round(0.2*self.phantomSize)):
            decayedRotatedMatrix = decay(rotatedMatrix, self.T2, 0.008)
            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
                sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd

            rotatedMatrix = recovery(decayedRotatedMatrix,self.T1,0.1)

            self.showKSpace(kSpace)
            print(i)
            if i % 2 == 0:
                vectors = rotateX(rotatedMatrix, -1 * 20 * 2)
            else:
                vectors = rotateX(rotatedMatrix, 20 * 2)
        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)


    def artifact2(self):
        kSpace = np.zeros((self.phantomSize, self.phantomSize), dtype=np.complex_)
        vectors = np.zeros((self.phantomSize, self.phantomSize, 3))
        vectors[:, :, 2] = 1

        if self.cycles is None :
            self.cycles = 3
            vectors = self.startup(vectors) 
            self.cycles = None
        else :
            vectors = self.startup(vectors)
        
        vectors = rotateX (vectors, 60)

        for i in range(0, round(self.phantomSize)):
            rotatedMatrix = rotateX(vectors, 90)
            decayedRotatedMatrix = decay(rotatedMatrix, self.T2, 1)
            if i == int(self.phantomSize/2):
                threshold = int(self.phantomSize*0.15)
                self.T1[:,0:self.phantomSize-threshold] = self.T1[:,threshold:self.phantomSize]
                self.T2[:,0:self.phantomSize-threshold] = self.T2[:,threshold:self.phantomSize]
                self.T1[:,self.phantomSize-threshold:self.phantomSize] = 0
                self.T2[:,self.phantomSize-threshold:self.phantomSize] = 0
                decayedRotatedMatrix[:,0:self.phantomSize-threshold,:] = decayedRotatedMatrix[:,threshold:self.phantomSize,:]
                decayedRotatedMatrix[:, self.phantomSize-threshold:self.phantomSize, :] = 0
            for j in range(0, self.phantomSize):
                stepX = (360 / self.phantomSize) * i
                stepY = (360 / self.phantomSize) * j
                phaseEncodedMatrix = gradientXY(decayedRotatedMatrix, stepY, stepX)
                sigmaX = np.sum(phaseEncodedMatrix[:, :, 0])
                sigmaY = np.sum(phaseEncodedMatrix[:, :, 1])
                valueToAdd = np.complex(sigmaX, sigmaY)
                kSpace[i, j] = valueToAdd

            decayedRotatedMatrix[:, :, 0] = 0
            decayedRotatedMatrix[:, :, 1] = 0

            self.showKSpace(kSpace)
            print(i)
            vectors = recovery(decayedRotatedMatrix, self.T1, 10)
        kSpace = np.fft.fft2(kSpace)
        self.showReconstructedImage(kSpace)


    def showKSpace(self, img):
        img = img[:]
        # img = np.abs(img)
        img = 20 * np.log(np.abs(img))
        qimg = qimage2ndarray.array2qimage(np.abs(img))
        self.ui.kspaceLbl.setPixmap(QPixmap(qimg))

    def showReconstructedImage(self, img):
        img = img[:]
        img = np.abs(img)
        img = img - np.min(img)
        img = img * (255 / np.max(img))
        img = np.round(img)
        self.KsapceImage=deepcopy(img)
        qimg = qimage2ndarray.array2qimage(np.abs(img))
        self.ui.kspaceLbl.setPixmap(QPixmap(qimg))

    def showGraphicalRep(self):
        if self.prep == 'T2prep' and self.acquisition== 'gre':
            try:
                val = int(self.prepText)
            except:
                val = 0
            fs = 100 # sampling rate in Hz
            tau = 1
            t = np.arange(-20,20,1/fs)
            y = ss.rect(t-1,tau)
            ## draw gradients
            self.ui.graphicsView.setRange(xRange=[0,10])
            self.ui.graphicsView.setRange(yRange=[0,9])
            self.ui.graphicsView.plot(t+1.5*val,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+1+1.5*val,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2+1.5*val,y+2,pen=self.Pen1[2])
            tau2 = .15
            d = ss.rect(t-1,tau2)
            # draw RFs 
            self.ui.graphicsView.plot(t,d+8)
            self.ui.graphicsView.plot(t+val,-d+8)
            self.ui.graphicsView.plot(t+1.5*val,(self.FA/90)*d+8)
        elif self.prep == 'T2prep' and self.acquisition == 'ssfp':
            try:
                val = int(self.prepText)
                TE = int(self.TE*1000)
            except:
                val = 0
                TE = 1
            fs = 100 # sampling rate in Hz
            tau = 1
            t = np.arange(-20,20,1/fs)
            y = ss.rect(t-1,tau)
            ## draw gradients
            self.ui.graphicsView.setRange(xRange=[0,10])
            self.ui.graphicsView.setRange(yRange=[0,9])
            self.ui.graphicsView.plot(t+1.5*val,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+TE+1.5*val,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2*TE+1.5*val,y+2,pen=self.Pen1[2])
            tau2 = .15
            d = ss.rect(t-1,tau2)
            # draw RFs 
            self.ui.graphicsView.plot(t,d+8)
            self.ui.graphicsView.plot(t+val,-d+8)
            self.ui.graphicsView.plot(t+1.5*val,(self.FA/90)*d+8)
            self.ui.graphicsView.plot(t+1.5*val+2.5*TE,(2*self.FA/90)*d+8)
        elif self.prep == 'T2prep' and self.acquisition == 'se':
            try:
                val = int(self.prepText)
                TE = int(self.TE*1000)
            except:
                val = 0
                TE = 1
            fs = 100 # sampling rate in Hz
            tau = 1
            t = np.arange(-20,20,1/fs)
            y = ss.rect(t-1,tau)
            ## draw gradients
            self.ui.graphicsView.setRange(xRange=[0,10])
            self.ui.graphicsView.setRange(yRange=[0,9])
            self.ui.graphicsView.plot(t+1.5*val,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+TE+1.5*val,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2*TE+1.5*val,y+2,pen=self.Pen1[2])
            x = 2*TE + 1.5*val
            self.ui.graphicsView.plot(t+x+1,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+x+2,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+x+3,y+2,pen=self.Pen1[2])
            tau2 = .15
            d = ss.rect(t-1,tau2)
            # draw RFs 
            self.ui.graphicsView.plot(t,d+8)
            self.ui.graphicsView.plot(t+val,-d+8)
            self.ui.graphicsView.plot(t+1.5*val,(self.FA/90)*d+8)
            self.ui.graphicsView.plot(t+1.5*val+3*TE,(2*self.FA/90)*d+8)
        elif self.prep == 'tagging' and self.acquisition == 'gre':
            fs = 100 # sampling rate in Hz
            tau = 1
            t = np.arange(-20,20,1/fs)
            y = ss.rect(t-1,tau)
            ## draw gradients
            self.ui.graphicsView.setRange(xRange=[0,10])
            self.ui.graphicsView.setRange(yRange=[0,9])
            self.ui.graphicsView.plot(t+1.5,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+1+1.5,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2+1.5,y+2,pen=self.Pen1[2])
            tau2 = .15
            d = ss.rect(t-1,tau2)
            # draw RFs 
            d2 = ss.rect(t-1,0.1)
            for i in range(10):
                self.ui.graphicsView.plot(t+i/10,d2+8)
            self.ui.graphicsView.plot(t+1.5,(self.FA/90)*d+8)
        elif self.prep == 'tagging' and self.acquisition == 'ssfp':
            fs = 100    # sampling rate in Hz
            tau = 1
            t = np.arange(-20,20,1/fs)
            y = ss.rect(t-1,tau)
            ## draw gradients
            self.ui.graphicsView.setRange(xRange=[0,10])
            self.ui.graphicsView.setRange(yRange=[0,9])
            self.ui.graphicsView.plot(t+1.5,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+1+1.5,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2+1.5,y+2,pen=self.Pen1[2])
            tau2 = .15
            d = ss.rect(t-1,tau2)
            # draw RFs 
            d2 = ss.rect(t-1,0.1)
            for i in range(10):
                self.ui.graphicsView.plot(t+i/10,d2+8)
            self.ui.graphicsView.plot(t+1.5,(self.FA/90)*d+8)
            self.ui.graphicsView.plot(t+1.5+2.5,(2*self.FA/90)*d+8)
        elif self.prep == 'tagging' and self.acquisition == 'se':
            fs = 100    # sampling rate in Hz
            tau = 1
            t = np.arange(-20,20,1/fs)
            y = ss.rect(t-1,tau)
            ## draw gradients
            self.ui.graphicsView.setRange(xRange=[0,10])
            self.ui.graphicsView.setRange(yRange=[0,9])
            self.ui.graphicsView.plot(t+1.5,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+1+1.5,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2+1.5,y+2,pen=self.Pen1[2])
            self.ui.graphicsView.plot(t+2.5+2,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+2.5+3,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2.5+4,y+2,pen=self.Pen1[2])
            tau2 = .15
            d = ss.rect(t-1,tau2)
            # draw RFs 
            d2 = ss.rect(t-1,0.1)
            for i in range(10):
                self.ui.graphicsView.plot(t+i/10,d2+8)
            self.ui.graphicsView.plot(t+1.5,(self.FA/90)*d+8)
            self.ui.graphicsView.plot(t+1.5+3,(2*self.FA/90)*d+8)
        if self.prep == 'ir' and self.acquisition== 'gre':
            fs = 100 # sampling rate in Hz
            tau = 1
            t = np.arange(-20,20,1/fs)
            y = ss.rect(t-1,tau)
            ## draw gradients
            self.ui.graphicsView.setRange(xRange=[0,10])
            self.ui.graphicsView.setRange(yRange=[0,9])
            self.ui.graphicsView.plot(t+1.5,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+1+1.5,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2+1.5,y+2,pen=self.Pen1[2])
            tau2 = .15
            d = ss.rect(t-1,tau2)
            # draw RFs 
            self.ui.graphicsView.plot(t,2*d+8)
            self.ui.graphicsView.plot(t+1.5,(self.FA/90)*d+8)
        if self.prep == 'ir' and self.acquisition== 'ssfp':
            fs = 100 # sampling rate in Hz
            tau = 1
            t = np.arange(-20,20,1/fs)
            y = ss.rect(t-1,tau)
            ## draw gradients
            self.ui.graphicsView.setRange(xRange=[0,10])
            self.ui.graphicsView.setRange(yRange=[0,9])
            self.ui.graphicsView.plot(t+1.5,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+1+1.5,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2+1.5,y+2,pen=self.Pen1[2])
            tau2 = .15
            d = ss.rect(t-1,tau2)
            # draw RFs 
            self.ui.graphicsView.plot(t,2*d+8)
            self.ui.graphicsView.plot(t+1.5,(self.FA/90)*d+8)
            self.ui.graphicsView.plot(t+1.5+2.5,(2*self.FA/90)*d+8)
        if self.prep == 'ir' and self.acquisition== 'se':
            fs = 100 # sampling rate in Hz
            tau = 1
            t = np.arange(-20,20,1/fs)
            y = ss.rect(t-1,tau)
            ## draw gradients
            self.ui.graphicsView.setRange(xRange=[0,10])
            self.ui.graphicsView.setRange(yRange=[0,9])
            self.ui.graphicsView.plot(t+1.5,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+1+1.5,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2+1.5,y+2,pen=self.Pen1[2])
            self.ui.graphicsView.plot(t+2.5+2,y+6,pen=self.Pen1[0])
            self.ui.graphicsView.plot(t+2.5+3,y+4,pen=self.Pen1[1])
            self.ui.graphicsView.plot(t+2.5+4,y+2,pen=self.Pen1[2])
            tau2 = .15
            d = ss.rect(t-1,tau2)
            # draw RFs 
            self.ui.graphicsView.plot(t,2*d+8)
            self.ui.graphicsView.plot(t+1.5,(self.FA/90)*d+8)
            self.ui.graphicsView.plot(t+1.5+3,(2*self.FA/90)*d+8)


    def clear_graphicalRep(self):
        self.ui.graphicsView.clear()

    def error(self, message):
        errorBox = QMessageBox()
        errorBox.setIcon(QMessageBox.Warning)
        errorBox.setWindowTitle('WARNING')
        errorBox.setText(message)
        errorBox.setStandardButtons(QMessageBox.Ok)
        errorBox.exec_()

    def zoomIn(self):
        self.zoom=self.zoom+1
        if self.zoom==self.phantomSize:
          QMessageBox.question(self, 'Error', "No More ZoomIn allowed", QMessageBox.Ok)
        if self.zoomtype == 'PD' :
           self.img=self.PD[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T1' :
            #self.zoom=0
            self.img=self.T1[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T2' :
           # self.zoom=0
            self.img=self.T2[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]      
        self.showPhantomImage()

    def zoomOut(self):
        self.zoom=self.zoom-1
        if self.zoom==0:
            QMessageBox.question(self, 'Error', "No More ZoomOut allowed", QMessageBox.Ok)
        if self.zoomtype == 'PD' :
            self.img=self.PD[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T1' :
            self.img=self.T1[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T2' :
            self.img=self.T2[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]      
        self.showPhantomImage()
    def right(self):
        self.drag_y=self.drag_y+1
        if self.phantomSize-self.zoom+self.drag_y ==self.phantomSize:
            QMessageBox.question(self, 'Error', "No More Drag is allowed", QMessageBox.Ok)
   
        if self.zoomtype == 'PD' :
            self.img=self.PD[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T1' :
            self.img=self.T1[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T2' :
            self.img=self.T2[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]      
        self.showPhantomImage()

    def left(self):
        self.drag_y= self.drag_y -1
        if self.zoom+self.drag_y == 0:
              QMessageBox.question(self, 'Error', "No More Drag is allowed", QMessageBox.Ok)
        if self.zoomtype == 'PD' :
           self.img=self.PD[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T1' :
            self.img=self.T1[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T2' :
            self.img=self.T2[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]      
        self.showPhantomImage()   

    def up(self):
        self.drag_x=self.drag_x-1
        if self.zoom+self.drag_x == 0:
            QMessageBox.question(self, 'Error', "No More Drag is allowed", QMessageBox.Ok)
        if self.zoomtype == 'PD' :
            self.img=self.PD[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T1' :
            self.img=self.T1[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T2' :
            self.img=self.T2[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]      
        self.showPhantomImage()  
    def down(self):
        self.drag_x=self.drag_x+1
        if self.phantomSize-self.zoom+self.drag_x ==self.phantomSize:
            QMessageBox.question(self, 'Error', "No More Drag is allowed", QMessageBox.Ok)
        if self.zoomtype == 'PD' :
            self.img=self.PD[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T1' :
            self.img=self.T1[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T2' :
            self.img=self.T2[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]      
        self.showPhantomImage()

    def link(self):
        if self.zoomtype == 'PD' :
            self.img=self.PD[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T1' :
            self.img=self.T1[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]
        elif self.zoomtype == 'T2' :
            self.img=self.T2[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y]      
        self.showPhantomImage()
        
        self.showReconstructedImage(self.KsapceImage[self.zoom+self.drag_x:self.phantomSize-self.zoom+self.drag_x,self.zoom+self.drag_y:self.phantomSize-self.zoom+self.drag_y])


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
