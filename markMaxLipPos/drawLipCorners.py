#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import os
import csv
import time

import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def CSVtoLIST(fileName):
    csvFILE = open(fileName, "r")
    csvObj = csv.reader(csvFILE)

    csvLIST = []
    for item in csvObj:
        csvLIST.append(item)
    csvLIST = np.array(csvLIST)

    return csvLIST

def drawLipCorner():
    '''
    將人工標示的連續數字8上唇下唇頂點畫在同一張 speaker 的 frame 上，顯示該 speaker 嘴巴的最大變動區間。
    '''
    lipCornerPos = CSVtoLIST("./opticalFlow/lipCorners_upperAndLower/LXCP_R_81139_upperAndLowerLip.csv")
    lipCornerPos = np.array(lipCornerPos,dtype='uint16')
    
    cap = cv2.VideoCapture("./MP4Files/LXCP_R_81139.MP4")
    
    frameID = 0
    while True:
        ret, frame = cap.read()
        if frame == None:
            break
        else:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            for corner in lipCornerPos:
                cv2.circle(frame, (corner[1], corner[2]), 2, [0,255,0], -1)
                cv2.circle(frame, (corner[3], corner[4]), 2, [0,0,255], -1)
            
            cv2.imshow("test",frame)
            #cv2.waitKey(0)
            #cv2.imwrite("LXCP_maxLipMove.jpg",frame)
            time.sleep(0.02)
        frameID += 1
    
    cv2.destroyAllWindows()

def drawRatioLineChart():
    '''
    畫出 speaker 嘴巴的最大位移量及臉寬（高）的比值折線圖、統計直方圖、平均值及標準差。
    '''
    moveRatio = []
    
    lipMaxMoveDic = {}
    faceWidthLIST = CSVtoLIST("./manMade_face_boundingBox.csv")
    for i, item in enumerate(faceWidthLIST):
        speaker = item[0][:4]
        faceWidth = int(item[3])-int(item[1])
        lipMaxMoveDic[speaker] = faceWidth
        print item
    lipMaxMove = CSVtoLIST("./opticalFlow/lipCorners_upperAndLower/maxLipMove.csv")
    speakerLIST = []
    ratio = []
    for i, item in enumerate(lipMaxMove):
        speaker = item[0]
        speakerLIST.append(speaker)
        if int(item[1]) >= int(item[2]):
            maxMoveNum = int(item[1])
        else:
            maxMoveNum = int(item[2])
        ratio.append(float(maxMoveNum)/lipMaxMoveDic["{}".format(item[0])])
        moveRatio.append([speaker, float(maxMoveNum)/lipMaxMoveDic["{}".format(item[0])]])
    ratio = np.array(ratio)
    ratioMean = round(np.mean(ratio),4)
    ratioStd = round(np.std(ratio),4)
    
    outputFILE = open("maxLipMoveRatio_faceWidth.csv", "w")
    w = csv.writer(outputFILE)
    w.writerows(moveRatio)
    outputFILE.close()
    os.system("mv maxLipMoveRatio_faceWidth.csv ./opticalFlow")
    
    x_axis = np.arange(1,len(speakerLIST)+1)
    
    font = FontProperties()
    alignment = {'horizontalalignment': 'center', 'verticalalignment': 'top'}
    font.set_weight('normal')
    font.set_family('monospace')
    font.set_size('small')
    font.set_style('normal')
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=False)
    
    fig.subplots_adjust(hspace=.5)
    # ax0 變動大小比例折線圖
    plot0, = ax0.plot(x_axis,ratio,color = 'blue',linewidth=0.5)
    ax0.set_xticks(x_axis)
    ax0.set_xticklabels(speakerLIST,rotation='vertical',**alignment)
    for label in ax0.get_xticklabels():
        label.set_fontproperties(font)
    ax0.set_title("Lip max movement / Face width in pixels")
    
    # ax1 變動大小比例統計直方圖
    binsSet = np.arange(0,max(ratio),0.01)
    ax1.hist(ratio,bins = binsSet)
    ax1.set_title("The histogram of variance ratio")
    ax1.text(0.18, 7, 'mean: {}'.format(ratioMean), style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
    ax1.text(0.18, 5.3, 'std: {}'.format(ratioStd), style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
    plt.savefig('lipMoveRatio_faceWidth.png',bbox_inches='tight')
    os.system("mv lipMoveRatio_faceWidth.png ./opticalFlow")
    #plt.show()

def markLeftEyePoint(event,x,y,flags,param):   # 標左眼中心的位置
    global leftEyeX,leftEyeY
    if event == cv2.EVENT_LBUTTONDOWN:
        leftEyeX,leftEyeY = x,y
        print "left ",leftEyeX,leftEyeY
    
def markRightEyePoint(event,x,y,flags,param):   # 標右眼中心的位置
    global rightEyeX,rightEyeY
    if event == cv2.EVENT_LBUTTONDOWN:
        rightEyeX,rightEyeY = x,y
        print "right ",rightEyeX,rightEyeY

def drawRatioMouthToEye():
    '''
    畫出嘴巴最大位移量及眼睛到臉底部這兩區段的比值折線圖、統計值方圖、平均值及標準差。
    '''
    moveRatio = []
    eyePosLIST = []
    
    # 取得各 speaker 眼睛到臉底部的高
    eyeToFaceBoundDic = {}
    faceHeightLIST = CSVtoLIST("./manMade_face_boundingBox.csv")
    for i, item in enumerate(faceHeightLIST):
        speaker = item[0][:4]
        kind = item[0][5]
        if kind == "R":
            kind = "Reading"
        else:
            kind = "Speaking"
        
        cap = cv2.VideoCapture("./MP4Files/{}".format(item[0]))
        ret, frame = cap.read()
        ix, iy, fx, fy = np.array(item[1:],dtype='uint16')
        
        # 取得眼睛位置
        leftEyeX = 0
        leftEyeY = 0
        rightEyeX = 0
        rightEyeY = 0
                
        cv2.namedWindow('mark left eye location')
        cv2.resizeWindow('mark left eye location', len(frame[0]), len(frame))
        cv2.setMouseCallback('mark left eye location',markLeftEyePoint)   # 以 mouse 拖移來控制 ROI 區域
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.imshow('mark left eye location', frame)
        cv2.destroyAllWindows()
        
        cv2.namedWindow('mark right eye location')
        cv2.resizeWindow('mark right eye location', len(frame[0]), len(frame))
        cv2.setMouseCallback('mark right eye location',markRightEyePoint)   # 以 mouse 拖移來控制 ROI 區域
        while True:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.imshow('mark right eye location', frame)
        cv2.destroyAllWindows()
        
        eyePosLIST.append([item[0],leftEyeX,leftEyeY,rightEyeX,rightEyeY])
        
        # 計算眼睛到臉底部的距離
        eyeToFaceBound = fy - (leftEyeY+rightEyeY)/2
        eyeToFaceBoundDic[speaker] = eyeToFaceBound
    
    outputFILE = open("manMadeEyePos.csv", "w")
    w = csv.writer(outputFILE)
    w.writerows(eyePosLIST)
    outputFILE.close()
    
    
    lipMaxMove = CSVtoLIST("./opticalFlow/lipCorners_upperAndLower/maxLipMove.csv")
    speakerLIST = []
    ratio = []
    for i, item in enumerate(lipMaxMove):
        speaker = item[0]
        speakerLIST.append(speaker)
        if int(item[1]) >= int(item[2]):
            maxMoveNum = int(item[1])
        else:
            maxMoveNum = int(item[2])
        ratio.append(float(maxMoveNum)/eyeToFaceBoundDic["{}".format(item[0])])
        moveRatio.append([speaker, float(maxMoveNum)/eyeToFaceBoundDic["{}".format(item[0])]])
    ratio = np.array(ratio)
    ratioMean = round(np.mean(ratio),4)
    ratioStd = round(np.std(ratio),4)

    outputFILE = open("maxLipMoveRatio_eye2face.csv", "w")
    w = csv.writer(outputFILE)
    w.writerows(moveRatio)
    outputFILE.close()
    os.system("mv maxLipMoveRatio_eye2face.csv ./opticalFlow")

    x_axis = np.arange(1,len(speakerLIST)+1)

    font = FontProperties()
    alignment = {'horizontalalignment': 'center', 'verticalalignment': 'top'}
    font.set_weight('normal')
    font.set_family('monospace')
    font.set_size('small')
    font.set_style('normal')
    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=False)

    fig.subplots_adjust(hspace=.5)
    # ax0 變動大小比例折線圖
    plot0, = ax0.plot(x_axis,ratio,color = 'blue',linewidth=0.5)
    ax0.set_xticks(x_axis)
    ax0.set_xticklabels(speakerLIST,rotation='vertical',**alignment)
    for label in ax0.get_xticklabels():
        label.set_fontproperties(font)
    ax0.set_title("Lip max movement / eye2faceBottom in pixels")

    # ax1 變動大小比例統計直方圖
    binsSet = np.arange(0,max(ratio),0.01)
    ax1.hist(ratio,bins = binsSet)
    ax1.set_title("The histogram of variance ratio")
    ax1.text(0.22, 7, 'mean: {}'.format(ratioMean), style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
    ax1.text(0.22, 5.3, 'std: {}'.format(ratioStd), style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
    plt.savefig('lipMoveRatio_eye2face.png',bbox_inches='tight')
    os.system("mv lipMoveRatio_eye2face.png ./opticalFlow")
    #plt.show()



if __name__ == "__main__":
    
    drawRatioLineChart()