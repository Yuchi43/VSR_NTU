#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import csv
import os
import time

import cv2


'''
目標：計算每個 speaker 嘴唇變化的最大量。
方法：以人工的方式標出 40 個 speaker 在念數字 8 時，上唇頂點及下唇頂點原始位置，紀錄該 frameID 及座標點。
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
        os.system("mv {}_upperAndLowerLip.csv ./opticalFlow/lipCorners_upperAndLower".format(csvDICT[stepLog]["MP4FILE"][:-4]))
        
        # 計算 speaker 上唇頂點 row 的最差距 及 下唇頂點 row 的最差距, 儲存在 maxMoveLIST 中
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
    os.system("mv maxLipMove.csv ./opticalFlow/lipCorners_upperAndLower")