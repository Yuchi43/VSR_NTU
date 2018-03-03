#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import csv
import os
import time

import cv2


'''
這支程式是輔助工具，能協助人工標出 40 個 speaker 在念數字 8 時，上唇頂點及下唇頂點的位置。

目標：取得每個 speaker 嘴唇變化的最大量。
方法：以人工的方式標出 40 個 speaker 在念數字 8 時，上唇頂點及下唇頂點原始位置，紀錄該 frameID 及座標點。
操作說明：程式執行後，首先播放第一部影片的第 1 張 frame， 視窗上顯示「mark upper lip corner」，用滑鼠點擊上唇頂點的位置，點擊後程式會將該點的 x, y 紀錄下來，
        不滿意點擊的位置可以重複點擊，程式會更新儲存的參數，滿意後點擊按鍵「q」可以換成點擊下唇頂點位置，視窗顯示 「mark lower lip corner」，操作與點擊上唇頂點相同。
        第 1 張 frame 完成後會接著播放第 4 張 frame (每 3 張標記一次頂點位置)，完成一部影片的標記後，會存出一個檔案 (見註2 第1點)。
        
        40 部影片都完成後，程式會存出每個 speaker 上下唇變化的最大量 (見註2 第2點)。


註1：影片取得方式詳見 VSR_NTU 中的 README.md。
註2：這支程式最後會存出兩個類型的檔案，分別是 
    1. {影片名稱}_upperAndLowerLip.csv     紀錄每三張 frame 標記一次上下唇頂點的位置資訊 (每個 speaker 的影片會存出一個 csv 檔)
                                         [格式]
                                         frameID, upperX, upperY, lowerX, lowerY
                                          
    2. maxLipMove.csv                    彙整 40 個 speaker 上下唇變動的最大量
                                         [格式]
                                         影片名稱, upperMaxMove, lowerMaxMove
'''

def CSVtoDICT(csvFILE):
    '''
    csvDICT scheme >>>
    csvDICT = { 1 : {"order"       : "1",             # 該影片在整個資料庫中的編號，在此實驗中沒有使用此參數。
                     "MP4FILE"     : "MP4FILE name",  # 影片檔名
                    },
                2 : {"order"       : "2",
                     "MP4FILE"     :"MP4FILE name",
                    },
                3...
              }
    '''
    csvDICT = {}
    csvFILE = open(csvFILE, "r")
    seperator = csv.Sniffer().sniff(csvFILE.read(1024), delimiters = ",;\t")
    csvFILE.seek(0)
    csvObj = csv.reader(csvFILE, seperator)
    stepLog = 0
    for i in csvObj:
        csvDICT[stepLog] = {"order"        : i[0],
                            "MP4FILE"      : i[1],
                            }
        stepLog += 1
    return csvDICT

def CSVtoLIST(fileName):
    csvFILE = open(fileName, "r")
    csvObj = csv.reader(csvFILE)

    csvLIST = []
    for item in csvObj:
        csvLIST.append(item)
    csvLIST = np.array(csvLIST)

    return csvLIST


def markUpperPoint(event,x,y,flags,param):   # 標上唇頂點的位置
    global upperX,upperY
    if event == cv2.EVENT_LBUTTONDOWN:
        upperX,upperY = x,y
        #print "mark:{}  {}".format(upperX,upperY)

def markLowerPoint(event,x,y,flags,param):   # 標上唇頂點的位置
    global lowerX,lowerY
    if event == cv2.EVENT_LBUTTONDOWN:
        lowerX,lowerY = x,y
        #print "mark2:{}  {}".format(lowerX,lowerY)


if __name__ == "__main__":
    
    fileName = 'testSheetShort_8contain.csv'
    try:
        csvDICT = CSVtoDICT("./{}".format(fileName))
        for key in csvDICT.keys():
            if os.path.exists("./MP4Files/"+csvDICT[key]["MP4FILE"]):
                pass
            else:
                raise SystemExit
        stepLog = 0
    except:
        raise SystemExit
    
    endFrameID = CSVtoLIST("eightEndFrameID.csv")[0]
    endFrameID = np.array(endFrameID,dtype='uint16')
    
    lipPointPosition = []      # 存 frameID 及 上下唇頂點的位置
    maxMoveLIST = []     # 計算各 speaker 最大 row 的差異
    
    while stepLog < len(csvDICT.keys()):
        
        cap = cv2.VideoCapture("./MP4Files/"+csvDICT[stepLog]["MP4FILE"])
        
        upperX = 0
        upperY = 0
        lowerX = 0
        lowerY = 0
        
        upperLipRowLIST = []
        lowerLipRowLIST = []
        singleSpeakerLipBound = []
        frameID = 0
        while frameID <= endFrameID[stepLog]:
            ret, frame = cap.read()
            
            if not (frameID % 3):
                
                cv2.namedWindow('mark upper lip corner: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]),1)
                cv2.resizeWindow('mark upper lip corner: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]), len(frame[0]), len(frame))
                cv2.setMouseCallback('mark upper lip corner: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]),markUpperPoint)   # 以 mouse 拖移來控制 ROI 區域
                while True:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    cv2.imshow('mark upper lip corner: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]), frame)
                cv2.destroyAllWindows()
                
                cv2.namedWindow('mark lower lip corner: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]),1)
                cv2.resizeWindow('mark lower lip corner: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]), len(frame[0]), len(frame))
                cv2.setMouseCallback('mark lower lip corner: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]),markLowerPoint)   # 以 mouse 拖移來控制 ROI 區域
                while True:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    cv2.imshow('mark lower lip corner: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]), frame)
                cv2.destroyAllWindows()
                
                #print "--------------"
                #print 'mark3:{}  {}  {}  {}'.format(upperX,upperY,lowerX,lowerY)
                singleSpeakerLipBound.append([frameID,upperX,upperY,lowerX,lowerY])
                
                upperLipRowLIST.append(upperY)
                lowerLipRowLIST.append(lowerY)
            
            
            frameID += 1
            #print frameID
        
        # 儲存 speaker 確切的上下唇頂點
        outputFILE = open("{}_upperAndLowerLip.csv".format(csvDICT[stepLog]["MP4FILE"][:-4]), "w")
        w = csv.writer(outputFILE)
        w.writerows(singleSpeakerLipBound)
        outputFILE.close()
        os.system("mv {}_upperAndLowerLip.csv ./result/lipCorners_upperAndLower".format(csvDICT[stepLog]["MP4FILE"][:-4]))
        
        # 計算 speaker 上唇頂點 row 的最大差距 及 下唇頂點 row 的最大差距, 儲存在 maxMoveLIST 中
        upperLipRowLIST.sort()
        lowerLipRowLIST.sort()
        
        upperMaxMove = upperLipRowLIST[len(upperLipRowLIST)-1] - upperLipRowLIST[0]
        lowerMaxMove = lowerLipRowLIST[len(lowerLipRowLIST)-1] - lowerLipRowLIST[0]
        maxMoveLIST.append([csvDICT[stepLog]["MP4FILE"][:4],upperMaxMove,lowerMaxMove])
        
        stepLog += 1
        
    outputFILE = open("maxLipMove.csv", "w")
    w = csv.writer(outputFILE)
    w.writerows(maxMoveLIST)
    outputFILE.close()
    os.system("mv maxLipMove.csv ./result/lipCorners_upperAndLower")