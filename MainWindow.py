import sys
import cv2
import numpy as np
import tensorflow as tf
import time
from PyQt5 import QtCore, QtGui, QtWidgets

# Load your trained model
model_path = 'model.h5'
model = tf.keras.models.load_model(model_path)

# Label map
label_map = {0: 'Cut', 1: 'Dressing', 2: 'F_Body', 3: 'Red_T'}

# Preallocate memory for input_data
input_data = np.empty((1, 224, 224, 3), dtype=np.float32)

def process_frame(frame, model, input_data):
    resized_frame = cv2.resize(frame, input_data.shape[1:3])
    np.copyto(input_data, resized_frame[np.newaxis] / 255.0)
    predictions = model.predict(input_data)
    
    top_index = np.argmax(predictions)
    top_label = label_map[top_index]
    top_prob = predictions[0][top_index]
    predictions[0][top_index] = 0
    
    second_index = np.argmax(predictions)
    second_label = label_map[second_index]
    second_prob = predictions[0][second_index]
    
    return [(top_label, top_prob), (second_label, second_prob)]


class Ui_MainWindow(object):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)

        # Set up the user interface from the generated file
        self.setupUi(self)

        # Connect signals and slots
        self.StartBtn.clicked.connect(self.start_detection)
        self.StopBtn.clicked.connect(self.stop_detection)
        self.Threshold.valueChanged.connect(self.set_threshold)

        # Initialize variables
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.prev_time = 0
        self.threshold = 100

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1082, 672)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.ROIFeed = QtWidgets.QLabel(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ROIFeed.sizePolicy().hasHeightForWidth())
        self.ROIFeed.setSizePolicy(sizePolicy)
        self.ROIFeed.setObjectName("ROIFeed")
        self.layoutWidget = QtWidgets.QWidget(self.splitter)
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.StartBtn = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StartBtn.sizePolicy().hasHeightForWidth())
        self.StartBtn.setSizePolicy(sizePolicy)
        self.StartBtn.setObjectName("StartBtn")
        self.verticalLayout.addWidget(self.StartBtn)
        self.StopBtn = QtWidgets.QPushButton(self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.StopBtn.sizePolicy().hasHeightForWidth())
        self.StopBtn.setSizePolicy(sizePolicy)
        self.StopBtn.setObjectName("StopBtn")
        self.verticalLayout.addWidget(self.StopBtn)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.Threshold = QtWidgets.QSlider(self.layoutWidget)
        self.Threshold.setProperty("value", 50)
        self.Threshold.setOrientation(QtCore.Qt.Horizontal)
        self.Threshold.setObjectName("Threshold")
        self.verticalLayout_2.addWidget(self.Threshold)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 3, 0, 1, 1)
        self.CameraFeed = QtWidgets.QLabel(self.centralwidget)
        self.CameraFeed.setObjectName("CameraFeed")
        self.gridLayout.addWidget(self.CameraFeed, 4, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1082, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.ROIFeed.setText(_translate("MainWindow", "TextLabel"))
        self.StartBtn.setText(_translate("MainWindow", "Start Detection"))
        self.StopBtn.setText(_translate("MainWindow", "Stop Detection"))
        self.label.setText(_translate("MainWindow", "Detection Threshold"))
        self.CameraFeed.setText(_translate("MainWindow", "TextLabel"))

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.timer.start(30)

    def stop_detection(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()

    def set_threshold(self, value):
        self.threshold = value

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time

        # Find the largest contour
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh, 400, 600, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = None
        for contour in contours:
            if largest_contour is None or cv2.contourArea(contour) > cv2.contourArea(largest_contour):
                largest_contour = contour

        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            max_side = max(w, h)
            margin = 20
            roi_x = max(0, x - (max_side - w) // 2 - margin)
            roi_y = max(0, y - (max_side - h) // 2 - margin)
            roi_w = min(max_side + 2 * margin, frame.shape[1] - roi_x)
            roi_h = min(max_side + 2 * margin, frame.shape[0] - roi_y)

            roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

            # Process the ROI and get the predictions
            top_two_preds = self.process_frame(roi)

            # Display the top two detected classes and confidences on the frame
            for i, (label, prob) in enumerate(top_two_preds):
                text = f"{label}: {prob * 100:.2f}%"
                cv2.putText(frame, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display FPS on the frame
            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame, fps_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Convert the frame to QImage and set it to the CameraFeed label
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        self.CameraFeed.setPixmap(QtGui.QPixmap.fromImage(qt_image))

        # Convert the ROI to QImage and set it to the ROIFeed label
        if largest_contour is not None:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            h, w, ch = roi.shape
            bytes_per_line = ch * w
            qt_roi = QtGui.QImage(roi.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
            self.ROIFeed.setPixmap(QtGui.QPixmap.fromImage(qt_roi))

        QtCore.QTimer.singleShot(1, self.update_frame)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = Ui_MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
