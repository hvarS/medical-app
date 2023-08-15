from qtpy import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
class calibrationSet(object):

    def __init__(self, Select, measured_length_in_pixels, image_shape) -> None:
        self.dialog = Select
        self.linpixels = measured_length_in_pixels
        Select.setWindowTitle("Recalibration")
        self.image_shape = image_shape
        self.custom_calibration = None

    def setupUi(self):
        
        self.dialog.setObjectName("Select")
        self.dialog.resize(500,400)
        self.groupBox = QtWidgets.QGroupBox(self.dialog)
        self.groupBox.setGeometry(QtCore.QRect(10, 20, 480, 160))
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(10, 10, 460, 30))
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("Segoe UI Variable Small Light")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(10, 50, 460, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Variable Small Light")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(10, 90, 460, 30))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Variable Small Light")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.groupBox_2 = QtWidgets.QGroupBox(self.dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(30, 210, 440, 100))
        self.groupBox_2.setObjectName("groupBox_2")
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.lineEdit.setGeometry(QtCore.QRect(340, 40, 100, 40))
        self.lineEdit.setObjectName("lineEdit")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setGeometry(QtCore.QRect(20, 40, 300, 30))
        self.label_4.setIndent(1)
        self.label_4.setObjectName("label_4")
        self.but = QtWidgets.QPushButton(self.dialog)
        self.but.setGeometry(QtCore.QRect(200, 350, 100, 40))
        self.but.setObjectName("calibrateButton")

        self.retranslateUi(self.dialog)
        self.map_clicks()
        # QtCore.QMetaObject.connectSlotsByName(self.dialog)

    def retranslateUi(self, Select):
        _translate = QtCore.QCoreApplication.translate
        Select.setWindowTitle(_translate("Select", "Re-Calibration"))
        self.groupBox.setTitle(_translate("Form", ""))
        self.label.setText(_translate("Form", f"Measured Length in Pixels: {self.linpixels}"))
        self.label_2.setText(_translate("Form", f"Image Height: {self.image_shape[0]} px"))
        self.label_3.setText(_translate("Form", f"Image Width : {self.image_shape[1]} px"))
        self.groupBox_2.setTitle(_translate("Form", ""))
        self.label_4.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt;\">Enter The Actual Length in µm</span></p></body></html>"))
        self.but.setText(_translate("Form","Calibrate"))

    
    def map_clicks(self):
        self.but.clicked.connect(self.clicked_calibrate)
        
    def clicked_calibrate(self):
        self.CALIBRATION_INDEX = 0
        self.custom_calibration = float(self.lineEdit.text())/float(self.linpixels)
        self.dialog.close()

    def popup(self):
        self.exec_()
        

