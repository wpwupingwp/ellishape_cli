#!/usr/bin/python3
from pathlib import Path
from sys import argv

import cv2
import numpy as np

from global_vars import log


def gui_chain_code_func(axis_info, origin_ori):
    nrow, ncol = axis_info.shape
    # print(nrow)
    # print(ncol)
    idxall = np.where(axis_info == 255)
    log.debug(f'{idxall=}')
    # idxall=np.transpose(idxall)
    # print(idxall)
    numoftotalpoints = len(idxall[0])

    rowidxst = origin_ori[0]
    colidxst = origin_ori[1]
    origin_ori = [rowidxst, colidxst]

    flags = np.zeros((nrow, ncol))
    contour_points = np.zeros((numoftotalpoints, 2))
    # print(contour_points)
    backword_points = np.zeros((numoftotalpoints, 2))
    chain_code_ori = np.zeros(numoftotalpoints, dtype=int)

    numofpoints = 0
    numofpoints_pre = 0
    contour_points[numofpoints] = origin_ori
    flags[rowidxst, colidxst] = 1
    numofbackword = 0
    backwordflag = False

    while numofpoints < numoftotalpoints - 1:
        fatecount = 8
        # print(numofpoints)
        rowidxpre, colidxpre = contour_points[numofpoints]  # 前一个点的坐标

        rowidx = int(min(max(rowidxpre, 0), nrow - 1))  #
        colidx = int(min(max(colidxpre + 1, 0), ncol - 1))
        # print(rowidx)
        # print(colidx)
        # print(rowidxst)
        # print(colidxst)

        if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
            numofpoints += 1
            contour_points[numofpoints] = [rowidx, colidx]
            chain_code_ori[numofpoints - 1] = 0
            break

        # print(axis_info[rowidx, colidx])
        # print(flags[rowidx, colidx])
        if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:
            numofpoints += 1
            # print(numofpoints)
            contour_points[numofpoints] = [rowidx, colidx]
            flags[rowidx, colidx] = 1
            chain_code_ori[numofpoints - 1] = 0
        else:
            fatecount -= 1
            rowidx = int(min(max(rowidxpre - 1, 0), nrow - 1))
            colidx = int(min(max(colidxpre + 1, 0), ncol - 1))
            if rowidx == origin_ori[0] and colidx == origin_ori[
                1] and numofpoints > 1:
                numofpoints += 1
                contour_points[numofpoints] = [rowidx, colidx]
                chain_code_ori[numofpoints - 1] = 1
                break
            # print(rowidx)
            # print(colidx)
            # print(axis_info[colidx,rowidx])
            # print(flags[rowidx, colidx])
            if axis_info[rowidx, colidx] == 255 and flags[rowidx, colidx] == 0:

                numofpoints += 1
                contour_points[numofpoints] = [rowidx, colidx]
                flags[rowidx, colidx] = 1
                chain_code_ori[numofpoints - 1] = 1
            else:
                fatecount -= 1
                rowidx = int(min(max(rowidxpre - 1, 0), nrow - 1))
                colidx = int(min(max(colidxpre, 0), ncol - 1))
                if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                    numofpoints += 1
                    contour_points[numofpoints] = [rowidx, colidx]
                    chain_code_ori[numofpoints - 1] = 2
                    break
                if axis_info[rowidx, colidx] == 255 and flags[
                    rowidx, colidx] == 0:
                    numofpoints += 1
                    contour_points[numofpoints] = [rowidx, colidx]
                    flags[rowidx, colidx] = 1
                    chain_code_ori[numofpoints - 1] = 2
                else:
                    fatecount -= 1
                    rowidx = int(min(max(rowidxpre - 1, 0), nrow - 1))
                    colidx = int(min(max(colidxpre - 1, 0), ncol - 1))
                    if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                        numofpoints += 1
                        contour_points[numofpoints] = [rowidx, colidx]
                        chain_code_ori[numofpoints - 1] = 3
                        break
                    if axis_info[rowidx, colidx] == 255 and flags[
                        rowidx, colidx] == 0:
                        numofpoints += 1
                        contour_points[numofpoints] = [rowidx, colidx]
                        flags[rowidx, colidx] = 1
                        chain_code_ori[numofpoints - 1] = 3
                    else:
                        fatecount -= 1
                        rowidx = int(min(max(rowidxpre, 0), nrow - 1))
                        colidx = int(min(max(colidxpre - 1, 0), ncol - 1))
                        if rowidx == rowidxst and colidx == colidxst and numofpoints > 2:
                            numofpoints += 1
                            contour_points[numofpoints] = [rowidx, colidx]
                            chain_code_ori[numofpoints - 1] = 4
                            break
                        if axis_info[rowidx, colidx] == 255 and flags[
                            rowidx, colidx] == 0:
                            numofpoints += 1
                            contour_points[numofpoints] = [rowidx, colidx]
                            flags[rowidx, colidx] = 1
                            chain_code_ori[numofpoints - 1] = 4
                        else:
                            fatecount -= 1
                            rowidx = int(min(max(rowidxpre + 1, 0), nrow - 1))
                            colidx = int(min(max(colidxpre - 1, 0), ncol - 1))
                            if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                                numofpoints += 1
                                contour_points[numofpoints] = [rowidx, colidx]
                                chain_code_ori[numofpoints - 1] = 5
                                break
                            if axis_info[rowidx, colidx] == 255 and flags[
                                rowidx, colidx] == 0:
                                numofpoints += 1
                                contour_points[numofpoints] = [rowidx, colidx]
                                flags[rowidx, colidx] = 1
                                chain_code_ori[numofpoints - 1] = 5
                            else:
                                fatecount -= 1
                                rowidx = int(
                                    min(max(rowidxpre + 1, 0), nrow - 1))
                                colidx = int(min(max(colidxpre, 0), ncol - 1))
                                if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                                    numofpoints += 1
                                    contour_points[numofpoints] = [rowidx,
                                                                   colidx]
                                    chain_code_ori[numofpoints - 1] = 6
                                    break
                                if axis_info[rowidx, colidx] == 255 and flags[
                                    rowidx, colidx] == 0:
                                    numofpoints += 1
                                    contour_points[numofpoints] = [rowidx,
                                                                   colidx]
                                    flags[rowidx, colidx] = 1
                                    chain_code_ori[numofpoints - 1] = 6
                                else:
                                    fatecount -= 1
                                    rowidx = int(
                                        min(max(rowidxpre + 1, 0), nrow - 1))
                                    colidx = int(
                                        min(max(colidxpre + 1, 0), ncol - 1))
                                    if rowidx == rowidxst and colidx == colidxst and numofpoints > 1:
                                        numofpoints += 1
                                        contour_points[numofpoints] = [rowidx,
                                                                       colidx]
                                        chain_code_ori[numofpoints - 1] = 7
                                        break
                                    if axis_info[rowidx, colidx] == 255 and \
                                            flags[rowidx, colidx] == 0:
                                        numofpoints += 1
                                        contour_points[numofpoints] = [rowidx,
                                                                       colidx]
                                        flags[rowidx, colidx] = 1
                                        chain_code_ori[numofpoints - 1] = 7
                                    else:
                                        fatecount -= 1

        if fatecount == 0:
            rowidxpre = int(contour_points[numofpoints, 0])
            colidxpre = int(contour_points[numofpoints, 1])
            # print(rowidxpre)
            # print(colidxpre)
            axis_info[rowidxpre, colidxpre] = 0
            if numofbackword == 0 and numofpoints > numofpoints_pre:
                numofpoints_pre = numofpoints
                backwordflag = 1
            numofpoints -= 1
            if numofpoints < 1:
                break
            numofbackword += 1
            if numofbackword > 20:
                break
            else:
                if backwordflag == 1:
                    backword_points[numofbackword] = [rowidxpre, colidxpre]
        else:
            numofbackword = 0
            backwordflag = False

    chain_code = chain_code_ori[:numofpoints]
    print(chain_code)
    # endpoint = backword_points[0] 
    oringin = origin_ori
    print(oringin)
    # print(endpoint)
    # print((np.where(flags == 1)))
    return chain_code, oringin


def calc_traversal_dist(ai):
    x_ = 0
    y_ = 0
    if(np.isscalar(ai)):
        p = np.zeros((1, 2))
        x_ += np.sign(6 - ai) * np.sign(2 - ai)
        y_ += np.sign(4 - ai) * np.sign(ai)
        p[0, 0] = x_
        p[0, 1] = y_
    else:
        p = np.zeros((ai.shape[0], 2))
        for i in range(ai.shape[0]):
            x_ += np.sign(6 - ai[i]) * np.sign(2 - ai[i])
            y_ += np.sign(4 - ai[i]) * np.sign(ai[i])
            p[i, 0] = x_
            p[i, 1] = y_
    return p


def is_completed_chain_code(chain_code, start_point):
    close_threshold = 2
    direction_vectors = np.array([
        # todo: right?
        [0, 1],  # 0: Up
        [-1, 1], # 1: Up-Left
        [-1, 0], # 2: Left
        [-1, -1],# 3: Down-Left
        [0, -1], # 4: Down
        [1, -1], # 5: Down-Right
        [1, 0],  # 6: Right
        [1, 1]   # 7: Up-Right
    ])
    end_point = np.array(start_point)
    for direction in chain_code:
        direction = direction % 8  # Ensure direction is within bounds
        end_point += direction_vectors[direction]
    distance = np.sqrt(np.sum((np.array(start_point) - end_point) ** 2))

    is_closed = (distance <= close_threshold)
    return is_closed, end_point


def chain_code(img_file: Path):
    # read color images and convert to gray
    img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
    img_result = np.zeros_like(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        log.error('Cannot find boundary in the image file')
        return None
    max_contour = max(contours, key=cv2.contourArea)
    max_contour = np.reshape(max_contour, (
    max_contour.shape[0], max_contour.shape[2]))
    log.debug(f'{max_contour[0][0]=}')
    log.info(f'Image size: {img.shape}')
    cv2.drawContours(img_result, [max_contour], -1, 255, thickness=1)
    # cv2.imshow('result_image',result_image)

    max_contour[:, [0, 1]] = max_contour[:, [1, 0]]
    boundary = max_contour
    chaincode, oringin = gui_chain_code_func(img_result, max_contour[0])
    log.debug(f'Chaincode shape: {chaincode.shape}')
    if len(chaincode) == 0:
        log.error('Cannot generate chain code from the image')
        return None

    log.debug(f'{boundary=}')
    # Draw green line for boundary
    cv2.polylines(img_result, boundary, isClosed=True, color=(0, 255, 0),
                  thickness=3)

    x_ = calc_traversal_dist(chaincode)
    x = np.vstack(([0, 0], x_))
    # print(x)

    # Draw red line for chain code traversal
    cv2.polylines(img_result, x, isClosed=True, color=(0, 0, 255),
                  thickness=3)
    is_closed, endpoint = is_completed_chain_code(chaincode, boundary[0])

    # todo: what is it?
    # self.graphicsView_4.fitInView(self.graphicsView_4.sceneRect(),
    #                               Qt.KeepAspectRatio)
    # # Get the current transform
    # transform = self.graphicsView_4.transform()
    # # Apply the scale transformation
    # transform.scale(7, 7)
    # self.graphicsView_4.setTransform(transform)
    # # Center the view on the specified point
    # self.graphicsView_4.centerOn(QPointF(endpoint[1], endpoint[0]))
    if not is_closed:
        log.error(f'Chain code is not closed')
        log.error(f'Chain code length: {chaincode.shape[0]}')
    else:
        log.debug(f'Chain code is closed')
        log.debug(f'{chaincode=}')
        log.debug(f'{chaincode.shape=}')
        # self.pushButton_9.setEnabled(True)
        # self.pushButton_10.setEnabled(True)
        # self.pushButton_17.setEnabled(True)
    EFA()
    return chaincode, img_file, id_full


class EFA():
    def __init__(self, chaincode, filename, id_full):
        super().__init__()
        self.setObjectName("MainWindow")
        self.resize(1143, 543)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.setFont(font)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox.setGeometry(QtCore.QRect(70, 100, 141, 51))
        self.checkBox.setObjectName("checkBox")
        self.checkBox.setChecked(True)
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(70, 150, 141, 51))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_2.setChecked(True)
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(70, 200, 141, 51))
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_3.setChecked(True)
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(70, 250, 141, 51))
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_4.setChecked(True)
        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(70, 300, 141, 51))
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_5.setChecked(True)
        self.checkBox_6 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_6.setGeometry(QtCore.QRect(70, 360, 141, 51))
        self.checkBox_6.setObjectName("checkBox_6")
        self.checkBox_6.setChecked(True)
        self.checkBox_7 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_7.setGeometry(QtCore.QRect(70, 420, 141, 51))
        self.checkBox_7.setObjectName("checkBox_7")
        self.checkBox_7.setChecked(True)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 80, 221, 401))
        self.groupBox.setObjectName("groupBox")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(250, 90, 421, 371))
        self.widget.setObjectName("widget")
        self.widget_3 = QtWidgets.QWidget(self.centralwidget)
        self.widget_3.setGeometry(QtCore.QRect(690, 90, 431, 371))
        self.widget_3.setObjectName("widget_3")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(310, 50, 221, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(310, 10, 231, 21))
        self.label_2.setObjectName("label_2")
        self.textEdit = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit.setGeometry(QtCore.QRect(550, 10, 81, 31))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(self.centralwidget)
        self.textEdit_2.setGeometry(QtCore.QRect(550, 50, 81, 31))
        self.textEdit_2.setObjectName("textEdit_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(700, 10, 201, 31))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(700, 50, 201, 31))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(910, 50, 201, 31))
        self.pushButton_3.setObjectName("pushButton_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(250, 470, 421, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(750, 470, 421, 16))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(250, 90, 421, 371))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView_2.setGeometry(QtCore.QRect(690, 90, 431, 371))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1143, 25))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("MainWindow", "EFA"))
        self.checkBox.setText(_translate("MainWindow", "Translatoin"))
        self.checkBox_2.setText(_translate("MainWindow", "Reverse"))
        self.checkBox_3.setText(_translate("MainWindow", "Scale"))
        self.checkBox_4.setText(_translate("MainWindow", "Rotate"))
        self.checkBox_5.setText(_translate("MainWindow", "Starting point"))
        self.checkBox_6.setText(_translate("MainWindow", "Y-symmetric"))
        self.checkBox_7.setText(_translate("MainWindow", "X-symmetric"))
        self.groupBox.setTitle(
            _translate("MainWindow", "normalization options"))
        self.label.setText(
            _translate("MainWindow", "Visualized harmonic number:"))
        self.label_2.setText(
            _translate("MainWindow", "Maximum harmonic number:"))
        self.pushButton.setText(_translate("MainWindow", "Calculate and save "))
        self.pushButton_2.setText(_translate("MainWindow", "Plot curve"))
        self.pushButton_3.setText(_translate("MainWindow", "Save curve"))
        self.textEdit.setText("35")
        self.textEdit_2.setText("35")
        self.label_3.setText(_translate("self",
                                        "reconstructed shapes using EFDs (red line) and chain-code (bhue line)"))
        self.label_4.setText(
            _translate("self", "reconstructed shapes using normalized EFDs"))

        self.scene = QGraphicsScene()
        self.scene_1 = QGraphicsScene()
        self.chaincode = chaincode
        self.id_full = id_full
        self.filename = filename

        self.checkBox_4.stateChanged.connect(self.onCheckBox4StateChanged)
        self.checkBox_5.stateChanged.connect(self.onCheckBox4StateChanged)
        self.pushButton.clicked.connect(self.cal_hs)
        self.pushButton_2.clicked.connect(self.plot_hs)
        self.pushButton_3.clicked.connect(self.save_hs)

    def onCheckBox4StateChanged(self):
        # Assuming you have imported Tkinter as tk
        # and you have a Tkinter variable handles
        t_1 = self.checkBox_4.isChecked()
        t_2 = self.checkBox_5.isChecked()

        if t_1 == 1 and t_2 == 1:
            self.checkBox_6.setEnabled(True)
            self.checkBox_7.setEnabled(True)
        else:
            self.checkBox_6.setChecked(False)
            self.checkBox_7.setChecked(False)
            self.checkBox_6.setEnabled(False)
            self.checkBox_7.setEnabled(False)

    def cal_hs(self):
        circle = self.textEdit_2.toPlainText()
        numofharmoinc = int(circle)

        ro = self.checkBox_4.isChecked()
        sc = self.checkBox_3.isChecked()
        re = self.checkBox_2.isChecked()
        y_sy = self.checkBox_6.isChecked()
        x_sy = self.checkBox_7.isChecked()
        sta = self.checkBox_5.isChecked()
        trans = self.checkBox.isChecked()
        option = [ro, sc, re, y_sy, x_sy, sta, trans]

        print(self.chaincode)

        _, a, b, c, d = fourier_approx_norm_modify(self.chaincode,
                                                   numofharmoinc, 1000, 1, 0,
                                                   option)

        t = np.transpose([a, b, c, d])
        Hs = np.reshape(t, (1, -1))
        coffs = [["filepath"]]
        cols = numofharmoinc * 4
        matrix = []
        for col in range(1, cols + 1):
            letter = chr(ord('a') + (col - 1) % 4)
            number = str((col - 1) // 4 + 1)
            matrix.append(letter + number)

        coffs[0].extend(matrix)
        coffs.append([f"{self.filename[:-4]}_{self.id_full}"])
        coffs[1].extend(Hs.flatten().tolist())

        df = pd.DataFrame(coffs)
        # 或者使用 with 语句，确保在写入后关闭 workbook
        with pd.ExcelWriter(
                f"results/{self.filename[:-4]}_{self.id_full}_info.xlsx",
                engine='openpyxl', mode='a',
                if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name='Sheet2', index=False, header=False)
        QMessageBox.information(None, "Success", "  done!  ")

    def plot_hs(self):
        self.scene.clear()
        self.scene_1.clear()
        chain_code = self.chaincode
        max_numofharmoinc = int(self.textEdit.toPlainText())
        mode = 0

        if chain_code.size > 0:
            contour_points = np.array([0, 0])
            chain_points = code2axis(chain_code, contour_points)
            for i in range(len(chain_points) - 1):
                x1, y1 = chain_points[i, :]
                x2, y2 = chain_points[i + 1, :]
                line = QGraphicsLineItem(x1, y1, x2, y2)
                line.setPen(QPen(QColor(0, 0, 255)))
                self.scene.addItem(line)
                self.scene_1.setSceneRect(self.scene_1.itemsBoundingRect())

            circle = self.textEdit_2.toPlainText()
            numofharmoinc = int(circle)
            if numofharmoinc > max_numofharmoinc:
                QMessageBox.warning(self, 'message',
                                    f"The input must be less than {max_numofharmoinc}")
            else:
                ro = self.checkBox_4.isChecked()
                sc = self.checkBox_3.isChecked()
                re = self.checkBox_2.isChecked()
                y_sy = self.checkBox_6.isChecked()
                x_sy = self.checkBox_7.isChecked()
                sta = self.checkBox_5.isChecked()
                trans = self.checkBox.isChecked()
                option = [ro, sc, re, y_sy, x_sy, sta, trans]

                x_, _, _, _, _ = fourier_approx_norm_modify(chain_code,
                                                            numofharmoinc, 400,
                                                            0, mode, option)
                chain_points_approx = np.vstack((x_, x_[0, :]))
                # print(chain_points_approx)

                for i in range(len(chain_points_approx) - 1):
                    x1 = chain_points_approx[i, 0] + contour_points[0]
                    y1 = contour_points[1] - chain_points_approx[i, 1]

                    x2 = chain_points_approx[i + 1, 0] + contour_points[0]
                    y2 = contour_points[1] - chain_points_approx[i + 1, 1]

                    line = QGraphicsLineItem(y1, x1, y2, x2)
                    line.setPen(QPen(QColor(255, 0, 0)))
                    self.scene.addItem(line)
                self.graphicsView.setScene(self.scene)
                self.graphicsView.fitInView(self.graphicsView.sceneRect(),
                                            Qt.KeepAspectRatio)

                x_, _, _, _, _ = fourier_approx_norm_modify(chain_code,
                                                            numofharmoinc, 400,
                                                            1, mode, option)
                chain_points_approx = x_
                for i in range(len(chain_points_approx) - 1):
                    x1 = chain_points_approx[i, 0] + contour_points[0]
                    y1 = contour_points[1] - chain_points_approx[i, 1]

                    x2 = chain_points_approx[i + 1, 0] + contour_points[0]
                    y2 = contour_points[1] - chain_points_approx[i + 1, 1]

                    line = QGraphicsLineItem(y1, x1, y2, x2)
                    line.setPen(QPen(QColor(255, 0, 0), 0.01))

                    self.scene_1.addItem(line)
                    self.scene_1.setSceneRect(self.scene_1.itemsBoundingRect())

                self.graphicsView_2.setScene(self.scene_1)
                self.graphicsView_2.fitInView(self.graphicsView_2.sceneRect(),
                                              Qt.KeepAspectRatio)

        else:
            print('Chain code is null')
            QMessageBox.warning(self, '', 'Chain code is null')

    def save_hs(self):
        options = QFileDialog.Options()
        defaultPath = './results/'
        defaultFileName = f"{defaultPath}{'_'.join(self.filename.split('.')[:-1])}_{self.id_full}.png"
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File",
                                                  defaultFileName,
                                                  "PNG files (*.png);;JPEG files (*.jpg);;All Files (*)",
                                                  options=options)

        if filePath:
            extension = filePath.split('.')[-1].lower()
            pixmap = QApplication.primaryScreen().grabWindow(
                self.winId())  # 截取整个屏幕或指定窗口
            if extension == 'png':
                pixmap.save(filePath + '.png')
            elif extension == 'jpg':
                pixmap.save(filePath + '.jpg')
            else:
                print('Unsupported file format. Image not saved.')

            print(f'Image saved as: {filePath}')


def code2axis(chain_code, start_point):
    end_point = start_point
    axis=np.zeros((len(chain_code)+1,2))
    axis[0,:]=start_point
    i=0
    for code in chain_code:
        i=i+1
        direction = code % 8
        if direction == 0:
            end_point = np.add(end_point , [0, 1])
        elif direction == 7:
            end_point = np.add(end_point , [1, 1])
        elif direction == 6:
            end_point = np.add(end_point , [1, 0])
        elif direction == 5:
            end_point = np.add(end_point , [1, -1])
        elif direction == 4:
            end_point = np.add(end_point , [0, -1])
        elif direction == 3:
            end_point = np.add(end_point , [-1, -1])
        elif direction == 2:
            end_point = np.add(end_point , [-1, 0])
        elif direction == 1:
            end_point = np.add(end_point , [-1, 1])
        # print(end_point)
        axis[i,:]=end_point
    return axis


def calc_traversal_time(ai):
    t_ = 0
    if np.isscalar(ai):
        t = np.zeros(1)
        t_ = t_ + 1 + ((np.sqrt(2) - 1) / 2) * (1 - (-1) ** ai)
        t[0] = t_
    else:
        t = np.zeros(ai.shape[0])
        for i in range(ai.shape[0]):
            t_ = t_ + 1 + ((np.sqrt(2) - 1) / 2) * (1 - (-1) ** ai[i])
            t[i] = t_
    return t


def calc_harmonic_coefficients_modify(ai, n, mode):
    k = ai.shape[0]
    d = calc_traversal_dist(ai)
    if mode == 0:
        edist = d[k - 1, 0] ** 2 + d[k - 1, 1] ** 2
        if edist > 2:
            print('error chaincode, not close form')
            return None
        else:
            if edist > 0:
                vect = [-d[k - 1, 0], -d[k - 1, 1]]
                if vect[0] == 1 and vect[1] == 0:
                    ai = np.append(ai, 0)
                elif vect[0] == 1 and vect[1] == 1:
                    ai = np.append(ai, 1)
                elif vect[0] == 0 and vect[1] == 1:
                    ai = np.append(ai, 2)
                elif vect[0] == -1 and vect[1] == 1:
                    ai = np.append(ai, 3)
                elif vect[0] == -1 and vect[1] == 0:
                    ai = np.append(ai, 4)
                elif vect[0] == -1 and vect[1] == -1:
                    ai = np.append(ai, 5)
                elif vect[0] == 0 and vect[1] == -1:
                    ai = np.append(ai, 6)
                elif vect[0] == 1 and vect[1] == -1:
                    ai = np.append(ai, 7)

    elif mode == 1:
        if d[k - 1, 0] != 0:
            exp1 = np.zeros(abs(d[k - 1, 0])) + 2 * np.sign(
                d[k - 1, 0]) + 2 * np.sign(d[k - 1, 0]) * np.sign(d[k - 1, 0])
            ai = np.append(ai, exp1)
        if d[k - 1, 1] != 0:
            exp2 = np.zeros(abs(d[k - 1, 1])) + 2 + 2 * np.sign(
                d[k - 1, 1]) + 2 * np.sign(d[k - 1, 1]) * np.sign(d[k - 1, 1])
            ai = np.append(ai, exp2)

    elif mode == 2:
        if d[k - 1, 0] != 0 or d[k - 1, 1] != 0:
            exp = (ai + 4) % 8
            ai = np.append(ai, exp)

    k = ai.shape[0]
    t = calc_traversal_time(ai)
    d = calc_traversal_dist(ai)
    T = t[k - 1]
    two_n_pi = 2 * n * np.pi

    sigma_a = 0
    sigma_b = 0
    sigma_c = 0
    sigma_d = 0

    for p in range(k):
        if p >= 1:
            tp_prev = t[p - 1]
            dp_prev = d[p - 1]
        else:
            tp_prev = 0
            dp_prev = np.zeros(2)

        delta_d = calc_traversal_dist(ai[p])
        # print(delta_d)
        delta_x = delta_d[:, 0]
        delta_y = delta_d[:, 1]
        delta_t = calc_traversal_time(ai[p])

        q_x = delta_x / delta_t
        q_y = delta_y / delta_t

        sigma_a += two_n_pi * (
                    d[p, 0] * np.sin(two_n_pi * t[p] / T) - dp_prev[0] * np.sin(
                two_n_pi * tp_prev / T)) / T
        sigma_a += q_x * (np.cos(two_n_pi * t[p] / T) - np.cos(
            two_n_pi * tp_prev / T))
        sigma_b -= two_n_pi * (
                    d[p, 0] * np.cos(two_n_pi * t[p] / T) - dp_prev[0] * np.cos(
                two_n_pi * tp_prev / T)) / T
        sigma_b += q_x * (np.sin(two_n_pi * t[p] / T) - np.sin(
            two_n_pi * tp_prev / T))
        sigma_c += two_n_pi * (
                    d[p, 1] * np.sin(two_n_pi * t[p] / T) - dp_prev[1] * np.sin(
                two_n_pi * tp_prev / T)) / T
        sigma_c += q_y * (np.cos(two_n_pi * t[p] / T) - np.cos(
            two_n_pi * tp_prev / T))
        sigma_d -= two_n_pi * (
                    d[p, 1] * np.cos(two_n_pi * t[p] / T) - dp_prev[1] * np.cos(
                two_n_pi * tp_prev / T)) / T
        sigma_d += q_y * (np.sin(two_n_pi * t[p] / T) - np.sin(
            two_n_pi * tp_prev / T))

    if n == 0:
        print("n must be non-zero")

    try:
        r = T / (2 * n ** 2 * np.pi ** 2)
    except ZeroDivisionError:
        r = np.inf

    a = r * sigma_a
    b = r * sigma_b
    c = r * sigma_c
    d = r * sigma_d

    return [a, b, c, d]


def calc_dc_components_modify(ai, mode):
    k = ai.shape[0]
    t = calc_traversal_time(ai)
    d = calc_traversal_dist(ai)
    Tk = t[k - 1]

    if mode == 0:
        edist = d[k - 1, 0] ** 2 + d[k - 1, 1] ** 2
        if edist > 2:
            print('error chaincode, not close form')
            return None, None, None, None
        else:
            if edist > 0:
                vect = [-d[k - 1, 0], -d[k - 1, 1]]
                if vect[0] == 1 and vect[1] == 0:
                    ai = np.append(ai, 0)
                elif vect[0] == 1 and vect[1] == 1:
                    ai = np.append(ai, 1)
                elif vect[0] == 0 and vect[1] == 1:
                    ai = np.append(ai, 2)
                elif vect[0] == -1 and vect[1] == 1:
                    ai = np.append(ai, 3)
                elif vect[0] == -1 and vect[1] == 0:
                    ai = np.append(ai, 4)
                elif vect[0] == -1 and vect[1] == -1:
                    ai = np.append(ai, 5)
                elif vect[0] == 0 and vect[1] == -1:
                    ai = np.append(ai, 6)
                elif vect[0] == 1 and vect[1] == -1:
                    ai = np.append(ai, 7)

    elif mode == 1:
        if d[k - 1, 0] != 0:
            exp1 = np.zeros(abs(d[k - 1, 0])) + 2 * np.sign(
                d[k - 1, 0]) + 2 * np.sign(d[k - 1, 0]) * np.sign(d[k - 1, 0])
            ai = np.append(ai, exp1)
        if d[-1, 1] != 0:
            exp2 = np.zeros(abs(d[k - 1, 1])) + 2 + 2 * np.sign(
                d[k - 1, 1]) + 2 * np.sign(d[k - 1, 1]) * np.sign(d[k - 1, 1])
            ai = np.append(ai, exp2)

    elif mode == 2:
        if d[k - 1, 0] != 0 or d[k - 1, 1] != 0:
            exp = (ai + 4) % 8
            ai = np.append(ai, exp)
    k = ai.shape[0]
    t = calc_traversal_time(ai)
    d = calc_traversal_dist(ai)
    T = t[k - 1]

    sum_a0 = 0
    sum_c0 = 0
    for p in range(1, k):
        if p >= 1:
            dp_prev = d[p - 1, :]
        else:
            dp_prev = np.zeros(2)
        delta_t = calc_traversal_time(ai[p])
        sum_a0 += (d[p, 0] + dp_prev[0]) * delta_t / 2
        sum_c0 += (d[p, 1] + dp_prev[1]) * delta_t / 2

    A0 = sum_a0 / T
    C0 = sum_c0 / T

    return A0, C0, Tk, T


def fourier_approx_norm_modify(ai, n, m, normalized, mode, option):
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    for i in range(n):
        # print(ai.shape)
        harmonic_coeff = calc_harmonic_coefficients_modify(ai, i + 1, 0)
        a[i] = harmonic_coeff[0]
        b[i] = harmonic_coeff[1]
        c[i] = harmonic_coeff[2]
        d[i] = harmonic_coeff[3]

    A0, C0, Tk, T = calc_dc_components_modify(ai, 0)
    # print(A0)
    # print(C0)
    # print(Tk)
    # print(T)

    # Normalization procedure
    if normalized:
        ro = option[0]
        sc = option[1]
        re = option[2]
        y_sy = option[3]
        x_sy = option[4]
        sta = option[5]
        trans = option[6]

        # Remove DC components
        if trans:
            A0 = 0
            C0 = 0

        if re:
            CrossProduct = a[0] * d[0] - c[0] * b[0]
            if CrossProduct < 0:
                b = -b
                d = -d

        tan_theta2 = 2 * (a[0] * b[0] + c[0] * d[0]) / (
                    a[0] ** 2 + c[0] ** 2 - b[0] ** 2 - d[0] ** 2)
        theta1 = 0.5 * np.arctan(tan_theta2)
        if theta1 < 0:
            theta1 += np.pi / 2
        sin_2theta = np.sin(2 * theta1)
        cos_2theta = np.cos(2 * theta1)
        cos_theta_square = (1 + cos_2theta) / 2
        sin_theta_square = (1 - cos_2theta) / 2

        axis_theta1 = (a[0] ** 2 + c[0] ** 2) * cos_theta_square + (
                    a[0] * b[0] + c[0] * d[0]) * sin_2theta + (
                                  b[0] ** 2 + d[0] ** 2) * sin_theta_square
        axis_theta2 = (a[0] ** 2 + c[0] ** 2) * sin_theta_square - (
                    a[0] * b[0] + c[0] * d[0]) * sin_2theta + (
                                  b[0] ** 2 + d[0] ** 2) * cos_theta_square

        # print(axis_theta1)
        # print(axis_theta2)

        if axis_theta1 < axis_theta2:
            theta1 += np.pi / 2

        costh1 = np.cos(theta1)
        sinth1 = np.sin(theta1)
        a_star_1 = costh1 * a[0] + sinth1 * b[0]
        c_star_1 = costh1 * c[0] + sinth1 * d[0]
        psi1 = np.arctan(c_star_1 / a_star_1)
        if psi1 < 0:
            psi1 += np.pi

        E = np.sqrt(a_star_1 ** 2 + c_star_1 ** 2)
        # print(E)

        if sc:
            a /= E
            b /= E
            c /= E
            d /= E

        cospsi1 = np.cos(psi1)
        sinpsi1 = np.sin(psi1)
        normalized_all = np.zeros((n, 4))

        if ro:
            for i in range(n):
                normalized = np.dot([[cospsi1, sinpsi1], [-sinpsi1, cospsi1]],
                                    [[a[i], b[i]], [c[i], d[i]]])
                # print(normalized.reshape(1, 4))
                normalized_all[i] = normalized.reshape(1, 4)
            a = normalized_all[:, 0]
            b = normalized_all[:, 1]
            c = normalized_all[:, 2]
            d = normalized_all[:, 3]

        normalized_all_1 = np.zeros((n, 4))
        print(theta1)

        if sta:
            for i in range(n):
                normalized_1 = np.dot([[a[i], b[i]], [c[i], d[i]]], [
                    [np.cos(theta1 * (i + 1)), -np.sin(theta1 * (i + 1))],
                    [np.sin(theta1 * (i + 1)), np.cos(theta1 * (i + 1))]])
                normalized_all_1[i, :] = normalized_1.reshape(1, 4)
                # print(normalized_1)
            a = normalized_all_1[:, 0]
            b = normalized_all_1[:, 1]
            c = normalized_all_1[:, 2]
            d = normalized_all_1[:, 3]

            if a[0] < 0:
                a = -a
                d = -d

        if y_sy:
            if a[1] < 0:
                for i in range(1, n):
                    signval = (-1) ** (((i + 1) % 2) + 1)
                    a[i] = signval * a[i]
                    d[i] = signval * d[i]
                    signval = (-1) ** ((i + 1) % 2)
                    b[i] = signval * b[i]
                    c[i] = signval * c[i]

        if x_sy:
            if b[1] < 0:
                b[1:] *= -1
                c[1:] *= -1

    # print(a)
    # print(b)
    # print(c)
    # print(d)
    output = np.zeros((m, 2))

    for t in range(m):
        x_ = 0.0
        y_ = 0.0
        for i in range(n):
            x_ += (a[i] * np.cos(2 * (i + 1) * np.pi * (t) * Tk / (m - 1) / T) +
                   b[i] * np.sin(2 * (i + 1) * np.pi * (t) * Tk / (m - 1) / T))
            y_ += (c[i] * np.cos(2 * (i + 1) * np.pi * (t) * Tk / (m - 1) / T) +
                   d[i] * np.sin(2 * (i + 1) * np.pi * (t) * Tk / (m - 1) / T))
        output[t, 0] = A0 + x_
        output[t, 1] = C0 + y_

    return output, a, b, c, d


def main():
    img_file = Path(argv[1])
    a = chain_code(img_file)
    b = EFA(a)


if __name__ == '__main__':
    main()
