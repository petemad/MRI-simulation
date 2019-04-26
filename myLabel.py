from PyQt5 import QtWidgets
from PyQt5 import QtWidgets, QtCore, QtGui
import math
from PyQt5.QtGui import QPixmap


class myLabel(QtWidgets.QLabel):

    def __init__(self, parent=None):
        super(myLabel, self).__init__(parent=parent)
        self.pixelsClicked = [(0, 0), (0, 0), (0, 0)]
        self.pixelSelector = 0
        self.phantomSize = 32

    def paintEvent(self, e):
        super().paintEvent(e)
        paint = QtGui.QPainter(self)
        paint.begin(self)

        for pixelSet in self.pixelsClicked:
            x = pixelSet[0]
            y = pixelSet[1]
            xt = self.frameGeometry().width()
            yt = self.frameGeometry().height()
            x = x * (xt / self.phantomSize)
            y = y * (yt / self.phantomSize)
            x = math.ceil(x)
            y = math.ceil(y)
            if self.pixelSelector == 0:
                pen = QtGui.QPen(QtCore.Qt.red)
            if self.pixelSelector == 1:
                pen = QtGui.QPen(QtCore.Qt.blue)
            if self.pixelSelector == 2:
                pen = QtGui.QPen(QtCore.Qt.yellow)
            pen.setWidth(1)
            paint.setPen(pen)
            # draw rectangle on painter
            paint.drawRect(x - 10, y - 10, 20, 20)

            self.pixelSelector += 1
            self.pixelSelector = self.pixelSelector % 3
        paint.end()
