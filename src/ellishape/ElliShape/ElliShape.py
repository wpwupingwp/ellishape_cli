# -*- coding: utf-8 -*-

#print改成log.info
#sam改成fastsam

import sys
import numpy as np
import os
import glob
import pandas as pd
import cv2  # type: ignore
import torch
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets,QtCore,QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QPainterPath, QPainter, QPen,QColor
from PyQt5.QtCore import Qt, QPointF, QPoint, QRectF, QThread, pyqtSignal


from segment_anything import SamPredictor, sam_model_registry
import argparse
import json
from typing import Any, Dict, List
# import openpyxl


sys.path.append('/functions')
from functions.inverted_colors_func import inverted_colors_func
from functions.grayscale_func import grayscale_func
from functions.enhancement_func import enhancement_func
from functions.binarize_func import binarize_func
from functions.corrosion_func import corrosion_func
from functions.dilation_func import dilation_func
from functions.gui_chain_code_func import gui_chain_code_func
from functions.contour_extraction_func import contour_extraction_func
from functions.cal_area_c import cal_area_c
from functions.is_completed_chain_code import is_completed_chain_code
from functions.calc_traversal_dist import calc_traversal_dist
from functions.write_Data_To_File import write_data_to_file
from functions.EFA import MainWindow_1 

class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene, *args, **kwargs):
        super(CustomGraphicsView, self).__init__(*args, **kwargs)
        self.setScene(scene)
        self.drawing_1 = False
        self.drawing_2 = False
        self.last_point = QPoint()

    def mousePressEvent(self, event):
        # if event.button() == Qt.LeftButton:
        # 保留第一个图像项，清除其余所有项
        items = self.scene().items()
        print(len(items))
        if len(items) > 1:
            for item in items[:-1]:
                # self.scene().removeItem(item)
                if  isinstance(item, QGraphicsLineItem):
                    pen = item.pen()
                    if pen.color() == QColor(255, 0, 0) or pen.color() == QColor(0, 255, 0):
                        self.scene().removeItem(item)
        pos = event.pos()
        pos_adjusted = self.mapToScene(pos).toPoint()
        self.last_point = pos_adjusted
        if event.button() == Qt.LeftButton:
            self.drawing_1 = True
        if event.button() == Qt.RightButton:
            self.drawing_2 = True
        print("Mouse Press at", pos_adjusted)


    def mouseMoveEvent(self, event):
        if self.drawing_1 or self.drawing_2:
            pos = event.pos()
            pos_adjusted = self.mapToScene(pos).toPoint()
            print("Mouse Move to", pos_adjusted)
            if self.drawing_1:
                pen = QPen(QColor(255, 255, 255), 1)  
            if self.drawing_2:
                pen = QPen(QColor(0, 0, 0), 1)                
            self.draw_line(self.last_point, pos_adjusted, pen)
            self.last_point = pos_adjusted

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing_1 = False
            self.drawing_2 = False
            print("Mouse Release")
            # self.save_scene_as_image()

    def draw_line(self, start_point, end_point,pen):
        line = self.scene().addLine(start_point.x(), start_point.y(), end_point.x(), end_point.y(),pen)
        print(f"Draw line from {start_point} to {end_point}")
        
# class ProgressWindow(QWidget):
#     def __init__(self):
#         super().__init__()

#         self.setWindowTitle('Progress Window')
#         self.setGeometry(150, 150, 300, 100)

#         self.progress = QProgressBar(self)
#         self.progress.setGeometry(30, 40, 250, 25)
#         self.progress.setMaximum(100)

#         layout = QVBoxLayout()
#         layout.addWidget(self.progress)
#         self.setLayout(layout)

#     def update_progress(self, value):
#         self.progress.setValue(value)

# class WorkerThread(QThread):
#     progress_updated = pyqtSignal(int)
#     task_finished = pyqtSignal()

#     def __init__(self, segment_func):
#         super().__init__()
#         self.segment_func = segment_func

#     def run(self):
#         self.segment_func(self.progress_updated)
#         self.task_finished.emit()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")
        self.resize(1795, 969)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.setFont(font)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(10, 10, 101, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(10, 60, 591, 661))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(610, 60, 481, 321))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_3.setGeometry(QtCore.QRect(610, 390, 481, 331))
        self.graphicsView_3.setObjectName("graphicsView_3")

        self.scene_3 = QGraphicsScene(self)
        self.graphicsView_4 = CustomGraphicsView(self.scene_3,self)
        self.graphicsView_4.setGeometry(QtCore.QRect(1100, 60, 681, 661))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox.setGeometry(QtCore.QRect(120, 11, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(340, 10, 71, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(420, 10, 71, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(18)
        font.setBold(True)
        font.setItalic(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(500, 10, 151, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_5.setGeometry(QtCore.QRect(790, 10, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_6.setGeometry(QtCore.QRect(660, 10, 121, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_7.setGeometry(QtCore.QRect(1480, 10, 161, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_8.setGeometry(QtCore.QRect(1300, 10, 171, 41))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setObjectName("pushButton_8")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 730, 841, 131))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_9.setGeometry(QtCore.QRect(10, 40, 141, 71))
        self.pushButton_9.setObjectName("pushButton_9")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setGeometry(QtCore.QRect(170, 40, 31, 31))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setGeometry(QtCore.QRect(270, 40, 31, 31))
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.textEdit = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit.setGeometry(QtCore.QRect(200, 40, 61, 31))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit_2.setGeometry(QtCore.QRect(200, 80, 61, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.textEdit_3 = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit_3.setGeometry(QtCore.QRect(300, 40, 61, 31))
        self.textEdit_3.setObjectName("textEdit_3")
        self.textEdit_4 = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit_4.setGeometry(QtCore.QRect(300, 80, 61, 31))
        self.textEdit_4.setObjectName("textEdit_4")
        self.textEdit_5 = QtWidgets.QTextEdit(self.groupBox)
        self.textEdit_5.setGeometry(QtCore.QRect(370, 70, 281, 41))
        self.textEdit_5.setObjectName("textEdit_5")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(670, 50, 51, 19))
        self.radioButton.setObjectName("radioButton")
        self.radioButton.setChecked(True)
        self.buttonGroup = QtWidgets.QButtonGroup(self)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.radioButton)
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(670, 80, 61, 19))
        self.radioButton_2.setObjectName("radioButton_2")
        self.buttonGroup.addButton(self.radioButton_2)
        self.pushButton_10 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_10.setGeometry(QtCore.QRect(740, 40, 91, 71))
        self.pushButton_10.setObjectName("pushButton_10")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(370, 40, 311, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(270, 80, 31, 31))
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(170, 80, 31, 31))
        self.label_7.setObjectName("label_7")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(879, 730, 901, 131))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_3.setGeometry(QtCore.QRect(20, 40, 211, 19))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_3.setChecked(True)
        self.buttonGroup_2 = QtWidgets.QButtonGroup(self)
        self.buttonGroup_2.setObjectName("buttonGroup_2")
        self.buttonGroup_2.addButton(self.radioButton_3)
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_4.setGeometry(QtCore.QRect(20, 90, 191, 19))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.radioButton_4.setFont(font)
        self.radioButton_4.setObjectName("radioButton_4")
        self.buttonGroup_2.addButton(self.radioButton_4)
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_11.setGeometry(QtCore.QRect(220, 30, 111, 41))
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_12.setGeometry(QtCore.QRect(342, 30, 101, 41))
        self.pushButton_12.setObjectName("pushButton_12")
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setGeometry(QtCore.QRect(460, 40, 41, 16))
        self.label_3.setObjectName("label_3")
        self.textEdit_6 = QtWidgets.QTextEdit(self.groupBox_2)
        self.textEdit_6.setGeometry(QtCore.QRect(500, 30, 71, 41))
        self.textEdit_6.setAutoFillBackground(False)
        self.textEdit_6.setMidLineWidth(0)
        self.textEdit_6.setObjectName("textEdit_6")
        self.pushButton_13 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_13.setGeometry(QtCore.QRect(600, 30, 141, 41))
        self.pushButton_13.setObjectName("pushButton_13")
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox_2)
        self.comboBox_2.setGeometry(QtCore.QRect(220, 90, 111, 31))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("canny")
        self.comboBox_2.addItem("sobel")
        self.comboBox_2.addItem("zerocross")
        self.comboBox_2.addItem("laplace")
        self.comboBox_2.addItem("Roberts")
        self.comboBox_2.addItem("Prewitt")

        self.horizontalSlider = QtWidgets.QSlider(self.groupBox_2)
        self.horizontalSlider.setGeometry(QtCore.QRect(350, 90, 191, 31))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.pushButton_14 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_14.setGeometry(QtCore.QRect(560, 80, 181, 41))
        self.pushButton_14.setObjectName("pushButton_14")
        self.pushButton_15 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_15.setGeometry(QtCore.QRect(750, 30, 141, 91))
        self.pushButton_15.setObjectName("pushButton_15")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(20, 859, 841, 71))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setGeometry(QtCore.QRect(50, 30, 51, 21))
        self.label_4.setObjectName("label_4")
        self.textEdit_7 = QtWidgets.QTextEdit(self.groupBox_3)
        self.textEdit_7.setGeometry(QtCore.QRect(110, 20, 311, 41))
        self.textEdit_7.setAutoFillBackground(False)
        self.textEdit_7.setMidLineWidth(0)
        self.textEdit_7.setObjectName("textEdit_7")
        self.pushButton_16 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_16.setGeometry(QtCore.QRect(440, 20, 391, 41))
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_17 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_17.setGeometry(QtCore.QRect(880, 870, 891, 51))
        self.pushButton_17.setIconSize(QtCore.QSize(30, 30))
        self.pushButton_17.setObjectName("pushButton_17")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1795, 25))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)



        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "ElliShape"))
        self.pushButton.setText(_translate("MainWindow", "Folder"))
        self.pushButton_2.setText(_translate("MainWindow", "→"))
        self.pushButton_3.setText(_translate("MainWindow", "←"))
        self.pushButton_4.setText(_translate("MainWindow", "Inverted Colors"))
        self.pushButton_5.setText(_translate("MainWindow", "Automated Segmentation"))
        self.pushButton_6.setText(_translate("MainWindow", "Polygon Tool"))
        self.pushButton_7.setText(_translate("MainWindow", "Image Enhancement"))
        self.pushButton_8.setText(_translate("MainWindow", "Grayscale Conversion"))
        self.groupBox.setTitle(_translate("MainWindow", "2 Size (area and circumference)"))
        self.pushButton_9.setText(_translate("MainWindow", "Click two points"))
        self.label.setText(_translate("MainWindow", "X1:"))
        self.label_2.setText(_translate("MainWindow", "Y1:"))
        self.textEdit_5.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial\',\'Arial\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.radioButton.setText(_translate("MainWindow", "mm"))
        self.radioButton_2.setText(_translate("MainWindow", "inch"))
        self.pushButton_10.setText(_translate("MainWindow", "Skip"))
        self.label_5.setText(_translate("MainWindow", "The actual distance between two points:"))
        self.label_6.setText(_translate("MainWindow", "Y2:"))
        self.label_7.setText(_translate("MainWindow", "X2:"))
        self.groupBox_2.setTitle(_translate("MainWindow", "1 Chain code generation"))
        self.radioButton_3.setText(_translate("MainWindow", "Image Binarization"))
        self.radioButton_4.setText(_translate("MainWindow", "Edge Detection"))
        self.pushButton_11.setText(_translate("MainWindow", "Binarization"))
        self.pushButton_12.setText(_translate("MainWindow", "Erosion"))
        self.label_3.setText(_translate("MainWindow", "Size:"))
        self.textEdit_6.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial\',\'Arial\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\';\">3</span></p></body></html>"))
        self.pushButton_13.setText(_translate("MainWindow", "Dilation "))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Select"))
        self.pushButton_14.setText(_translate("MainWindow", "Contour Extraction"))
        self.pushButton_15.setText(_translate("MainWindow", "Chain Code"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3 Save"))
        self.label_4.setText(_translate("MainWindow", "Tag:"))
        self.textEdit_7.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Arial\',\'Arial\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"center\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Arial\';\">1</span></p></body></html>"))
        self.pushButton_16.setText(_translate("MainWindow", "Chain codes, labeled images and size"))
        self.pushButton_17.setText(_translate("MainWindow", "Elliptic Fourier Analysis"))

        self.folder_path = ''
        self.cwd = os.getcwd() 
        self.image = []
        self.gray_img = []
        self.bin = []
        self.crop_result = []
        self.scene_pos=[0,0]
        self.scene_2=QGraphicsScene()
        # self.scene_5=QGraphicsScene()
        # self.input_point =[]
        self.polygon = []
        self.polygon_item = None
        
        self.scene = QGraphicsScene()
        
        # self.graphicsView_4.setScene(self.scene_3)
        
        

        
        self.flag_point = 0
        self.flag_btn_6 = 0
        self.points = []
        self.points_pre = []
        self.boundary = []
        self.chaincode = []
        self.last_point = QPoint()   
        self.drawing = False   
        self.last_point = [] 

        self.id_full = '1'
        self.filename = ''
        self.unit = 1


        for widget in self.centralWidget().findChildren(QPushButton):
            widget.setDisabled(True)
        self.pushButton.setEnabled(True)

        self.pushButton.clicked.connect(self.on_pushButton_clicked)
        self.comboBox.activated.connect(self.comboBox_Callback)
        self.pushButton_2.clicked.connect(self.Nextbtn_clicked)
        self.pushButton_3.clicked.connect(self.Previousbtn_clicked)
        
        self.pushButton_4.clicked.connect(self.Inverted_color)
        self.pushButton_5.clicked.connect(self.SAM_segment)
        # self.pushButton_5.clicked.connect(self.start_task)
        
        self.pushButton_6.clicked.connect(self.ROI_selection)
        self.pushButton_8.clicked.connect(self.grayscale)
        self.pushButton_7.clicked.connect(self.enhancement)
        self.buttonGroup.buttonClicked.connect(self.units_set)
        self.buttonGroup_2.buttonClicked.connect(self.method_set)
        self.pushButton_11.clicked.connect(self.binarization)
        self.pushButton_12.clicked.connect(self.corrosion)
        self.pushButton_13.clicked.connect(self.dilation)
        self.pushButton_15.clicked.connect(self.chain_code)
        self.pushButton_14.clicked.connect(self.functions_selection)
        self.pushButton_9.clicked.connect(self.clicked_point)
        self.pushButton_16.clicked.connect(self.save)
        self.pushButton_10.clicked.connect(self.skip)
        self.pushButton_17.clicked.connect(self.go_to_window2)

    def on_pushButton_clicked(self):
        try:
            self.folder_path = QFileDialog.getExistingDirectory(self, "select folder", self.cwd)             
            if self.folder_path:
                self.comboBox.clear()
                self.image_files = []
                for file_path in glob.glob(os.path.join(self.folder_path, "*")):
                    if file_path.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                        self.image_files.append(os.path.basename(file_path))
                self.comboBox.addItems(self.image_files)
                print(len(self.image_files))

            else:
                QMessageBox.warning(self, "Warning", "folder is empty")
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    

    def comboBox_Callback(self):
        try:
            for widget in self.centralWidget().findChildren(QPushButton):
                widget.setDisabled(True)
            self.pushButton.setEnabled(True)
            self.pushButton_2.setEnabled(True)
            self.pushButton_3.setEnabled(True) 

            self.polygon = []
            self.polygon_item = None
            self.filename = self.comboBox.currentText()
            self.c_index = self.comboBox.currentIndex()

            file_path = os.path.join(self.folder_path,self.filename)
            if self.filename:            
                self.image = QPixmap(file_path) 
                pixmap_item = QGraphicsPixmapItem(self.image)
                self.scene.clear()
                self.scene_2.clear()
                self.scene_3.clear()
                self.scene.addItem(pixmap_item)
                self.graphicsView.setScene(self.scene)
                self.graphicsView.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(True)
                self.pushButton_4.setEnabled(True)
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def Previousbtn_clicked(self):
        try:
            if self.c_index != 0:

                for widget in self.centralWidget().findChildren(QPushButton):
                    widget.setDisabled(True)
                self.pushButton.setEnabled(True)
                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(True) 

                self.c_index -= 1
                # print(self.c_index)
                self.filename = self.image_files[self.c_index]
                file_path = os.path.join(self.folder_path,self.filename)
                print(file_path)
                self.image = QPixmap(file_path)
                ncols = self.image.width()            
                nrows = self.image.height()

                pixmap_item = QGraphicsPixmapItem(self.image)

                self.scene.clear()
                self.scene_2.clear()
                self.scene_3.clear()
                self.scene.addItem(pixmap_item)
                self.comboBox.setCurrentIndex(self.c_index)
                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(True)
                self.pushButton_4.setEnabled(True)
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)
            else:
                QMessageBox.warning(self, "Warning", "This is the first one.")
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def Nextbtn_clicked(self):
        try:
            if self.c_index != len(self.image_files)-1:
                for widget in self.centralWidget().findChildren(QPushButton):
                    widget.setDisabled(True)
                self.pushButton.setEnabled(True)
                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(True) 
                self.c_index += 1
                # print(self.c_index)
                self.filename = self.image_files[self.c_index]
                file_path = os.path.join(self.folder_path,self.filename)
                # print(file_path)
                self.image = QPixmap(file_path)
                ncols = self.image.width()            
                nrows = self.image.height()

                pixmap_item = QGraphicsPixmapItem(self.image)

                self.scene.clear()
                self.scene_2.clear()
                self.scene_3.clear()
                self.scene.addItem(pixmap_item)
                self.comboBox.setCurrentIndex(self.c_index)
                self.pushButton_2.setEnabled(True)
                self.pushButton_3.setEnabled(True)
                self.pushButton_4.setEnabled(True)
                self.pushButton_5.setEnabled(True)
                self.pushButton_6.setEnabled(True)

            else:
                print("This is the last one")
                QMessageBox.warning(self, "Warning", "This is the last one.")
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def Inverted_color(self):
        try:
            if self.image:
                img_v = self.pixmap_to_numpy(self.image)
                img_inverted = inverted_colors_func(img_v)
                img_inverted = self.numpy_to_pixmap(img_inverted)
                self.image=img_inverted
                pixmap_item = QGraphicsPixmapItem(img_inverted)
                self.scene.clear()
                self.scene.addItem(pixmap_item)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def pixmap_to_numpy(self,pixmap):
        try:
            image = QImage(pixmap.toImage())
            print(image.format())  
            if image.format() in (QImage.Format_ARGB32, QImage.Format_RGB32, QImage.Format_ARGB32_Premultiplied):
                ptr = image.bits()
                ptr.setsize(image.byteCount())      
                arr = np.array(ptr).reshape(image.height(), image.width(), 4)   
                if (image.format() == QImage.Format_RGB32) or (image.format()== QImage.Format_ARGB32) :     
                    arr = arr[:, :, :3] 

                elif image.format()  == QImage.Format_ARGB32_Premultiplied:
                    # print(arr.shape)
                    # Handle premultiplied alpha
                    alpha = arr[:, :, 3:4] / 255.0
                    bgr = arr[:, :, :3] / alpha

                    # Clip values to ensure they are in valid range [0, 255]
                    bgr = np.clip(bgr, 0, 255)

                    # Convert to uint8 data type
                    bgr = bgr.astype(np.uint8)

                    # Convert from RGBA to BGR
                    arr = bgr[..., ::-1]

                    # Debugging: print the shape and dtype of the array
                print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")        
                return arr       
            # Handle QImage.Format_RGB888
            elif image.format()== QImage.Format_RGB888:
                width = image.width()
                height = image.height()
                ptr = image.bits()
                ptr.setsize(image.byteCount())
                
                # Convert to numpy array
                arr = np.array(ptr).reshape(height, width, 3)
                
                # Debugging: print the shape and dtype of the array
                print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")
                return arr


            else:
                raise ValueError("The image format does not support conversion to a numpy array.")
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def numpy_to_pixmap(self, numpy_array):
        try:
            if len(numpy_array.shape) == 2:  
                height, width = numpy_array.shape
                bytes_per_line = width
                q_image = QImage(numpy_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
                return QPixmap.fromImage(q_image)
            elif len(numpy_array.shape) == 3:  
                height, width, channel = numpy_array.shape
                bytes_per_line = channel * width
                if channel == 3:
                    q_image = QImage(numpy_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
                elif channel == 4:
                    q_image = QImage(numpy_array.data, width, height, bytes_per_line, QImage.Format_RGBA8888)
                else:
                    raise ValueError("Unsupported number of channels: {}".format(channel))
                return QPixmap.fromImage(q_image)
            else:
                raise ValueError("Unsupported image format: {}".format(numpy_array.shape))
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def show_mask(self, mask, random_color=False,i=0):
        try:
            if random_color:
                color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
            else:
                color = np.array([30/255, 144/255, 255/255, 0.6])
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

            # cv2.imshow('mask'+str(i),mask_image)
            mask_image =  (mask_image * 255).astype(np.uint8)
            # cv2.imshow('mask',mask_image)
            print(np.max(mask_image))
            img_mask = self.numpy_to_pixmap(mask_image)
            pixmap_item = QGraphicsPixmapItem(img_mask)
            self.scene.addItem(pixmap_item)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    # def start_task(self):
    #     try:
    #         self.progress_window = ProgressWindow()
    #         self.progress_window.show()

    #         self.worker_thread = WorkerThread(self.SAM_segment)

    #         self.worker_thread.progress_updated.connect(self.progress_window.update_progress)
    #         self.worker_thread.task_finished.connect(self.progress_window.close)

    #         self.worker_thread.start()
    #     except ValueError as e:
    #         QMessageBox.warning(self, 'Error', str(e))
    #         return

    def SAM_segment(self):#,progress_signal
        try:
            sam = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
            #安装软件的时候判断是否有gpu；选择下载哪个参数文件；数据文件跟代码分开；需要把参数文件放在数据中心
            #有就用_h;没有就用_l
            #先比较一下差异 几十张图测试一下时间
            #点改成框试试

            #整个打包放在pypi网站
            #qt打包可以压缩
            #不同参数，模型差异对比
            sam.to(device='cpu')
            predictor = SamPredictor(sam)
            self.filename = self.image_files[self.c_index]
            file_path = os.path.join(self.folder_path,self.filename)
            print(file_path)
            image = cv2.imread(file_path)
            QMessageBox.information(None, "Message", f"Running now, please wait a moment!")     
            # progress_signal.emit(40)  # 进度更新
            if image is not None:

                print("Image was successfully read.")
                predictor.set_image(image)
        
                x = self.scene_pos[0] 
                y = self.scene_pos[1] 
                print(x)
                print(y)
                input_point = np.array([[round(x),round(y)]])
                print(input_point)
                # progress_signal.emit(60)  # 进度更新
                masks, scores, logits = predictor.predict(
                        point_coords= input_point,
                        point_labels = np.array([1]),
                        multimask_output=False,
                    )               
                if(masks.shape[0]!=0):
                    self.pushButton_7.setEnabled(False)
                    self.pushButton_8.setEnabled(False)
                    self.pushButton_11.setEnabled(False)
                    self.pushButton_12.setEnabled(True)
                    self.pushButton_13.setEnabled(True)
                    self.pushButton_15.setEnabled(True)
                    for i, (mask, score) in enumerate(zip(masks, scores)):
                        if i==0:
                            self.show_mask(mask,True,i)
                            #将mask添加到3和4视图
                            self.bin=(mask* 255).astype(np.uint8)
                            pixmap_result = self.numpy_to_pixmap(self.bin)
                            pixmap_result_item = QGraphicsPixmapItem(pixmap_result)
                            self.scene_2.clear()
                            self.scene_3.clear()
                            self.scene_3.addItem(pixmap_result_item)    
                            self.graphicsView_3.setScene(self.scene_3)
                            self.graphicsView_3.fitInView(self.graphicsView_3.sceneRect(), Qt.KeepAspectRatio)
                            self.graphicsView_4.setScene(self.scene_3)
                            self.graphicsView_4.fitInView(self.graphicsView_4.sceneRect(), Qt.KeepAspectRatio)            
                            return 
                # progress_signal.emit(100)  # 进度更新
            else:
                # print("Failed to read image. Please check the file path.")
                QMessageBox.warning(self, "Warning", "Failed to read image. Please check the file path.")
                # progress_signal.emit(100)  # 进度更新
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return
        
    def wheelEvent(self, event):
        try:
            pos=event.pos()
            pos_adjusted = QtCore.QPoint(pos.x() - 10, pos.y() - 60)  
            if self.graphicsView.viewport().rect().contains(pos_adjusted):
                if(event.angleDelta().y()>0.5):
                    self.graphicsView.scale(1.5,1.5)
                elif(event.angleDelta().y()< 0.5):
                    self.graphicsView.scale(1 /1.5,1 / 1.5)

            pos_adjusted_2 = QtCore.QPoint(pos.x() - 610, pos.y() - 60)  
            if self.graphicsView_2.viewport().rect().contains(pos_adjusted_2):
                if(event.angleDelta().y()>0.5):
                    self.graphicsView_2.scale(1.5,1.5)
                elif(event.angleDelta().y()< 0.5):
                    self.graphicsView_2.scale(1 /1.5,1 / 1.5)

            pos_adjusted_3 = QtCore.QPoint(pos.x() - 610, pos.y() - 390) 
            if self.graphicsView_3.viewport().rect().contains(pos_adjusted_3):
                if(event.angleDelta().y()>0.5):
                    self.graphicsView_3.scale(1.5,1.5)
                elif(event.angleDelta().y()< 0.5):
                    self.graphicsView_3.scale(1 /1.5,1 / 1.5)

            pos_adjusted_4 = QtCore.QPoint(pos.x() - 1100, pos.y() - 60) 
            if self.graphicsView_4.viewport().rect().contains(pos_adjusted_4):
                if(event.angleDelta().y()>0.5):
                    self.graphicsView_4.scale(1.5,1.5)
                elif(event.angleDelta().y()< 0.5):
                    self.graphicsView_4.scale(1 /1.5,1 / 1.5)    
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def mousePressEvent(self, event):
        try:
            pos = event.pos()
            pos_adjusted = QtCore.QPoint(pos.x() - 10, pos.y() - 60) 
            scenepos = self.graphicsView.mapToScene(pos_adjusted)
            self.scene_pos[0]=scenepos.x()
            self.scene_pos[1]=scenepos.y()         
            if (self.flag_btn_6==1) & (event.buttons() == Qt.LeftButton) & (self.graphicsView.viewport().rect().contains(pos_adjusted)) :                    
                scenepos = self.graphicsView.mapToScene(pos_adjusted)
                self.scene_pos[0]=scenepos.x()
                self.scene_pos[1]=scenepos.y()
                print("Mouse Position (View Coordinates):", pos_adjusted.x(), pos_adjusted.y())
                print("Mouse Position (Scene Coordinates):", self.scene_pos[0],  self.scene_pos[1])
                self.polygon.append(scenepos)
                self.updatePolygon()
            elif (self.flag_btn_6==1) & (event.buttons() == Qt.RightButton) & (self.graphicsView.viewport().rect().contains(pos)) :
                pixmap_item=self.cropImage()          

                height=self.image.height()
                width=self.image.width()   
                mask = np.zeros((height, width), dtype=np.uint8)
                pts = np.array([(point.x(), point.y()) for point in self.polygon], dtype=np.int32)

                cv2.fillPoly(mask, [pts], 255)
                # print(np.shape(self.image))
                try:
                    img_num=self.pixmap_to_numpy(self.image)
                except ValueError as e:
                    QMessageBox.warning(self,'Error','{e}')
                    return
                result = cv2.bitwise_and(img_num, img_num, mask=mask)
                result[(result[:,:,0] == 0) & (result[:,:,1] == 0) & (result[:,:,2] == 0)] = [255, 255, 255]
                # cv2.imshow('result',result)
                self.crop_result =  cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                pixmap_result = self.numpy_to_pixmap(self.crop_result)
                pixmap_result_item = QGraphicsPixmapItem(pixmap_result)
                self.scene_2.clear()
                self.scene_2.addItem(pixmap_result_item)
                self.graphicsView_2.setScene(self.scene_2)
                self.graphicsView_2.fitInView(self.graphicsView_2.sceneRect(), Qt.KeepAspectRatio)
                self.polygon = []
                self.polygon_item = None
                self.flag_btn_6=0
                self.pushButton_8.setEnabled(True)

            if (self.flag_point == 1) & (event.button() == Qt.LeftButton) & (self.graphicsView.viewport().rect().contains(pos_adjusted)):           
                if hasattr(self, 'points'):
                    self.points_pre.append((event.x(), event.y()))
                else:
                    self.points_pre = [(event.x(), event.y())]
                print(self.points_pre)
                print(len(self.points_pre))
                if len(self.points_pre) == 2:
                    pos_dis1 = QtCore.QPoint(self.points_pre[0][0] - 10, self.points_pre[0][1] - 60) 
                    scenepos_dis1 = self.graphicsView.mapToScene(pos_dis1)
                    self.points.append((scenepos_dis1.x(),scenepos_dis1.y()))
                    pos_dis2 = QtCore.QPoint(self.points_pre[1][0] - 10, self.points_pre[1][1] - 60) 
                    scenepos_dis2 = self.graphicsView.mapToScene(pos_dis2)
                    self.points.append((scenepos_dis2.x(),scenepos_dis2.y()))
                    print(self.points)
                    self.distance = ((self.points[1][0] - self.points[0][0])**2 + (self.points[1][1] - self.points[0][1])**2)**0.5
                    self.textEdit.setText(str(int(self.points[0][0])))
                    self.textEdit_3.setText(str(int(self.points[0][1])))
                    self.textEdit_2.setText(str(int(self.points[1][0])))
                    self.textEdit_4.setText(str(int(self.points[1][1])))
                    point_item = QGraphicsEllipseItem(self.points[0][0]-10, self.points[0][1]-10, 20, 20) 
                    point_item.setBrush(Qt.red)  
                    self.scene.addItem(point_item)
                    point_item = QGraphicsEllipseItem(self.points[1][0]-10, self.points[1][1]-10, 20, 20)  
                    point_item.setBrush(Qt.red)  
                    self.scene.addItem(point_item) 

                    self.graphicsView.setScene(self.scene)
                    self.points_pre=[]
                    self.points = []
                    self.flag_point=0
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return
       

           

    def updatePolygon(self):
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
        if len(self.polygon) < 2:
            return

        path = QPainterPath()
        path.moveTo(self.polygon[0])
        for point in self.polygon[1:]:
            path.lineTo(point)
        path.lineTo(self.polygon[0])

        self.polygon_item = self.scene.addPath(path, QPen(Qt.red,3))


    def cropImage(self):
        if len(self.polygon) < 3:
            return

        path = QPainterPath()
        path.moveTo(self.polygon[0])
        for point in self.polygon[1:]:
            path.lineTo(point)
        path.lineTo(self.polygon[0])

        

        region = QImage(self.image.size(), QImage.Format_ARGB32)
        region.fill(Qt.transparent)

        painter = QPainter(region)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setClipPath(path)
        # painter.drawImage(0, 0, self.image)
        painter.end()

        cropped_pixmap = QPixmap.fromImage(region)
        pixmap_item = QGraphicsPixmapItem(cropped_pixmap)
        return pixmap_item

    def grayscale(self):
        try:
            self.gray_img = grayscale_func(self.crop_result)
            pixmap_result = self.numpy_to_pixmap(self.gray_img)
            pixmap_result_item = QGraphicsPixmapItem(pixmap_result)
            self.scene_2.clear()
            self.scene_2.addItem(pixmap_result_item)
            self.graphicsView_2.setScene(self.scene_2)
            self.graphicsView_2.fitInView(self.graphicsView_2.sceneRect(), Qt.KeepAspectRatio)
            self.pushButton_7.setEnabled(True)
            self.pushButton_11.setEnabled(True)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def enhancement(self):
        try:
            self.gray_img=enhancement_func(self.gray_img)
            pixmap_result = self.numpy_to_pixmap(self.gray_img)
            pixmap_result_item = QGraphicsPixmapItem(pixmap_result)
            self.scene_2.clear()
            self.scene_2.addItem(pixmap_result_item)
            self.graphicsView_2.setScene(self.scene_2)
            self.graphicsView_2.fitInView(self.graphicsView_2.sceneRect(), Qt.KeepAspectRatio)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def units_set(self):
        try:
            selectedButton =self.buttonGroup.checkedButton()
            id=selectedButton.text()
            # print(id)
            if (id=='mm'):
                self.unit=1
            elif (id=='inch'):
                self.unit=2
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def method_set(self):
        try:
            selectedButton =self.buttonGroup_2.checkedButton()
            id=selectedButton.text()
            # print(id)
            if (id=='Image Binarization'):
                self.pushButton_11.setEnabled(True)
                self.pushButton_12.setEnabled(True)
                self.pushButton_13.setEnabled(True)
                self.pushButton_14.setEnabled(False)
            elif (id=='Edge Detection'):
                self.pushButton_11.setEnabled(False)
                self.pushButton_12.setEnabled(False)
                self.pushButton_13.setEnabled(False)
                self.pushButton_14.setEnabled(True)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def binarization(self):
        try:
            self.bin = binarize_func(self.gray_img)
            self.bin = cv2.bitwise_not(self.bin)
            pixmap_result = self.numpy_to_pixmap(self.bin)
            pixmap_result_item = QGraphicsPixmapItem(pixmap_result)
            self.scene_3.clear()
            self.scene_3.addItem(pixmap_result_item)
            self.graphicsView_3.setScene(self.scene_3)
            self.graphicsView_3.fitInView(self.graphicsView_3.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView_4.setScene(self.scene_3)
            self.graphicsView_4.fitInView(self.graphicsView_4.sceneRect(), Qt.KeepAspectRatio)
            self.pushButton_12.setEnabled(True)
            self.pushButton_13.setEnabled(True)
            # self.pushButton_14.setEnabled(True)
            self.pushButton_15.setEnabled(True)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def corrosion(self):
        try:
            circle=self.textEdit_6.toPlainText()
            circle=int(circle)
            self.bin=corrosion_func(self.bin,circle)
            pixmap_result = self.numpy_to_pixmap(self.bin)
            pixmap_result_item = QGraphicsPixmapItem(pixmap_result)
            self.scene_3.clear()
            self.scene_3.addItem(pixmap_result_item)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return


    def dilation(self):
        try:
            self.bin = dilation_func(self.bin)
            pixmap_result = self.numpy_to_pixmap(self.bin)
            pixmap_result_item = QGraphicsPixmapItem(pixmap_result)
            self.scene_3.clear()
            self.scene_3.addItem(pixmap_result_item)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def chain_code(self):
        try:
            scene_rect = self.scene_3.sceneRect()
            pixmap = QPixmap(scene_rect.size().toSize())
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            self.scene_3.render(painter, QRectF(pixmap.rect()), scene_rect)
            painter.end()
            image = pixmap.toImage()
            image.save("scene_image.png")
            print("Image saved")
            
            self.bin = cv2.imread("scene_image.png",cv2.IMREAD_GRAYSCALE)
            # self.bin = self.bin[:,:,0]
            contours, _ = cv2.findContours(self.bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:

                max_contour = max(contours, key=cv2.contourArea)
                max_contour=np.reshape(max_contour, (max_contour.shape[0], max_contour.shape[2]))        
                # print(max_contour[0][0])
                result_image = np.zeros_like(self.bin)
                print(self.bin.shape)
                cv2.drawContours(result_image, [max_contour], -1, 255, thickness=1)
                # cv2.imshow('result_image',result_image)
                max_contour[:, [0, 1]] = max_contour[:, [1, 0]]
                self.boundary=max_contour
                self.chaincode,oringin = gui_chain_code_func(result_image,max_contour[0])
                print(self.chaincode.shape)

                # Check if chain_code is not empty
                if len(self.chaincode) == 0:
                    QMessageBox.warning(self, "Warning", "Chain code is empty")
                    return

                # Draw green line for boundary
                pen = QPen(Qt.green, 3)
                print(self.boundary)
                for i in range(len(self.boundary)-1):
                    start_point = QPointF(self.boundary[i][1], self.boundary[i][0])
                    end_point = QPointF(self.boundary[i + 1][1], self.boundary[i + 1][0])
                    line_item_1 = QGraphicsLineItem(start_point.x(), start_point.y(), end_point.x(), end_point.y())
                    line_item_1.setPen(pen)
                    self.scene_3.addItem(line_item_1)

                x_ = calc_traversal_dist(self.chaincode)
                x = np.vstack(([0, 0], x_))
                # print(x)

                # # Draw red line for chain code traversal
                pen.setColor(Qt.red)
                pen.setWidth(2)
                origin = self.boundary[0]
                current_point = QPointF(origin[1],origin[0])
                for move in x:
                    next_point = QPointF(origin[1] + move[0],origin[0]  - move[1])
                    print(next_point)
                    line_item = QGraphicsLineItem(current_point.x(), current_point.y(), next_point.x(), next_point.y())
                    line_item.setPen(pen)
                    self.scene_3.addItem(line_item)
                    current_point = next_point                    
                is_closed,endpoint = is_completed_chain_code(self.chaincode, self.boundary[0])
                self.graphicsView_4.fitInView(self.graphicsView_4.sceneRect(), Qt.KeepAspectRatio)
                # Get the current transform
                transform = self.graphicsView_4.transform()
                # Apply the scale transformation
                transform.scale(7, 7)
                self.graphicsView_4.setTransform(transform)
                # Center the view on the specified point
                self.graphicsView_4.centerOn(QPointF(endpoint[1],endpoint[0]))

                if not is_closed:
                    QMessageBox.critical(None, "Error", f"Chain code is not closed (length is {self.chaincode.shape[0]}), please edit.")

                else:
                    QMessageBox.information(None, "Success", f"Chain code is closed, and length is {self.chaincode.shape[0]}")               
                    print(self.chaincode)
                    print(self.chaincode.shape)
                    self.pushButton_9.setEnabled(True)
                    self.pushButton_10.setEnabled(True)
                    self.pushButton_17.setEnabled(True)
            else:
                print("not found boundary")
                QMessageBox.warning(self, "Warning", "not found boundary")
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def functions_selection(self):
        try:
            selected_item = self.comboBox_2.currentText()
            min_value = self.horizontalSlider.minimum()
            max_value = self.horizontalSlider.maximum()
            current_value = self.horizontalSlider.value()
            threshvalue = 255*(current_value-min_value)/(max_value-min_value)
            if selected_item=="canny":
                self.bin=contour_extraction_func(self.gray_img,1,threshvalue/2,threshvalue)
            elif selected_item=="sobel":
                self.bin=contour_extraction_func(self.gray_img,2,threshvalue/2,threshvalue)
            elif selected_item=="zerocross":
                self.bin=contour_extraction_func(self.gray_img,3,threshvalue/2,threshvalue)   
            elif selected_item=="laplace":
                self.bin=contour_extraction_func(self.gray_img,4,threshvalue/2,threshvalue)    
            elif selected_item=="Roberts":
                self.bin=contour_extraction_func(self.gray_img,5,threshvalue/2,threshvalue)  
            elif selected_item=="Prewitt":
                self.bin=contour_extraction_func(self.gray_img,6,threshvalue/2,threshvalue) 
            else:
                QMessageBox.warning(self, 'Error', 'Please select the an operator') 
                return

            pixmap_result = self.numpy_to_pixmap(self.bin)
            pixmap_result_item = QGraphicsPixmapItem(pixmap_result)
            self.scene_3.clear()
            self.scene_3.addItem(pixmap_result_item) 
            self.graphicsView_3.setScene(self.scene_3)
            self.graphicsView_3.fitInView(self.graphicsView_3.sceneRect(), Qt.KeepAspectRatio)
            self.graphicsView_4.setScene(self.scene_3)
            self.graphicsView_4.fitInView(self.graphicsView_4.sceneRect(), Qt.KeepAspectRatio) 
            self.pushButton_15.setEnabled(True)
        except ValueError as e:
            QMessageBox.warning(self,'Error','{e}')
            return

    def clicked_point(self):
        self.flag_point=1
        self.pushButton_16.setEnabled(True)

    def save(self):
        if self.textEdit_7.toPlainText():
            id_full = self.textEdit_7.toPlainText()
            self.id_full = id_full
            self.filename = self.image_files[self.c_index]

            if not os.path.exists('results'):
                os.makedirs('results')
            if not os.path.exists('label'):
                os.makedirs('label')
            # Writing boundary data to file
            boundary_filename = f"results/{self.filename[:-4]}_{id_full}_b.txt"

            write_data_to_file(self.boundary, boundary_filename)

            # Writing chain code data to file
            chain_filename = f"results/{self.filename[:-4]}_{id_full}_c.txt"
            write_data_to_file(self.chaincode, chain_filename)

            # Writing image to file
            image_filename = f"label/{self.filename[:-4]}_{id_full}.png"
            cv2.imwrite(image_filename,self.bin)
            
            try:
                # Processing distance data
                dis = float(self.textEdit_5.toPlainText())
                if dis>0:
                    dis_pixel = self.distance / dis
                    dis_mm = dis / self.distance
                else:
                    dis_pixel=0
                    dis_mm=0
                [area,circumference]=cal_area_c(self.chaincode,self.boundary)


                # Writing results to Excel file
                if self.unit==1:
                    results = {
                        'filepath': [f"{self.filename[:-4]}_{id_full}"],
                        'scale:pixels/mm': [dis_pixel],
                        'scale:mm/pixel': [dis_mm],
                        'circumference:pixel': [circumference],  
                        'area:pixel': [area],  # Example calculation
                        'circumference:mm': [circumference * dis_mm],  # Example calculation
                        'area:mm^2': [area * dis_mm ** 2],  # Example calculation
                    }
                elif self.unit==2:
                    results = {
                        'filepath': [f"{self.filename[:-4]}_{id_full}"],
                        'scale:pixels/inch': [dis_pixel],
                        'scale:inch/pixel': [dis_mm],
                        'circumference:pixel': [circumference],  
                        'area:pixel': [area],  # Example calculation
                        'circumference:inch': [circumference * dis_mm],  # Example calculation
                        'area:inch^2': [area * dis_mm ** 2],  # Example calculation
                    }
                df = pd.DataFrame(results)
                df.to_excel(f"results/{self.filename[:-4]}_{id_full}_info.xlsx", index=False,sheet_name='Sheet1')
                QMessageBox.information(None, "Success",'    done!   ')
            except Exception as e:
                QMessageBox.critical(None, "Error", str(e))
        else:
            print('Chain code is null. /Chain code is not closed.')
            QMessageBox.warning(self, 'Error', 'Chain code is null. /Chain code is not closed.')      
        
    def ROI_selection(self):
        self.flag_btn_6=1

    def go_to_window2(self):
        # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        self.window2 = MainWindow_1(self.chaincode,self.filename,self.id_full)
        self.window2.show()


    def skip(self):
        self.points.append((0,0))
        self.points.append((0,0))

        print(self.points)

        self.distance = 0

        self.textEdit.setText(str(0))
        self.textEdit_3.setText(str(0))
        self.textEdit_2.setText(str(0))
        self.textEdit_4.setText(str(0))
        self.textEdit_5.setText(str(0))
        self.pushButton_16.setEnabled(True)


if __name__ == '__main__':
    
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    MainWindow = MainWindow()
    MainWindow.show()
    sys.exit(app.exec_())
