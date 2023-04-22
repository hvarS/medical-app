from qtpy import QtCore, QtGui, QtWidgets

class calibrationSelection(object):

    def __init__(self, parent, Select) -> None:
        self.parent = parent
        self.dialog = Select
        self.CALIBRATION_INDEX = 2
        

    def setupUi(self):
        
        self.dialog.setObjectName("Select")
        self.dialog.resize(460,140)
        self.pushButton4x = QtWidgets.QPushButton(self.dialog)
        self.pushButton4x.setGeometry(QtCore.QRect(10, 20, 80, 40))
        self.pushButton4x.setObjectName("pushButton4x")
        self.pushButton10x = QtWidgets.QPushButton(self.dialog)
        self.pushButton10x.setGeometry(QtCore.QRect(100, 20, 80, 40))
        self.pushButton10x.setObjectName("pushButton10x")
        self.pushButton20x = QtWidgets.QPushButton(self.dialog)
        self.pushButton20x.setGeometry(QtCore.QRect(190, 20, 80, 40))
        self.pushButton20x.setObjectName("pushButton20x")
        self.pushButton40x = QtWidgets.QPushButton(self.dialog)
        self.pushButton40x.setGeometry(QtCore.QRect(280, 20, 80, 40))
        self.pushButton40x.setObjectName("pushButton40x")
        self.pushButton100x = QtWidgets.QPushButton(self.dialog)
        self.pushButton100x.setGeometry(QtCore.QRect(370, 20, 80, 40))
        self.pushButton100x.setObjectName("pushButton100x")

        self.recalButton =  QtWidgets.QPushButton(self.dialog)
        self.recalButton.setGeometry(QtCore.QRect(180,80,100,40))
        self.recalButton.setObjectName("recalButton")



        self.retranslateUi(self.dialog)
        self.map_clicks()
        QtCore.QMetaObject.connectSlotsByName(self.dialog)

    def retranslateUi(self, Select):
        _translate = QtCore.QCoreApplication.translate
        Select.setWindowTitle(_translate("Select", "Select Calibration Mode"))
        self.pushButton4x.setText(_translate("Select", "4x"))
        self.pushButton10x.setText(_translate("Select", "10x"))
        self.pushButton20x.setText(_translate("Select", "20x"))
        self.pushButton40x.setText(_translate("Select", "40x"))
        self.pushButton100x.setText(_translate("Select", "100x"))
        self.recalButton.setText(_translate("Select","Re-Calibrate"))

    
    def map_clicks(self):
        self.pushButton4x.clicked.connect(self.clicked_4x)
        self.pushButton10x.clicked.connect(self.clicked_10x)
        self.pushButton20x.clicked.connect(self.clicked_20x)
        self.pushButton40x.clicked.connect(self.clicked_40x)
        self.pushButton100x.clicked.connect(self.clicked_100x)
        self.recalButton.clicked.connect(self.clicked_recal)

    def clicked_recal(self):
        self.dialog.close()
        self.parent.mode = 'RECAL'
        self.parent.recalibrationMode()

    def clicked_4x(self):
        self.CALIBRATION_INDEX = 0
        self.dialog.close()


    def clicked_10x(self):
        self.CALIBRATION_INDEX = 1
        self.dialog.close()

    def clicked_20x(self):
        self.CALIBRATION_INDEX = 2
        self.dialog.close()
    
    def clicked_40x(self):
        self.CALIBRATION_INDEX = 3
        self.dialog.close()
    
    def clicked_100x(self):
        self.CALIBRATION_INDEX = 4
        self.dialog.close()

