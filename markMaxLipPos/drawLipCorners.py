#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import os
import csv
import time

import cv2
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

'''
這支是繪圖程式。將人工標記的上下唇位置畫在同一張 speaker 的影像上。

註： 這支程式只是為了觀察少量資料才撰寫的，因此讀檔及寫檔的檔名都是直接寫固定值，沒有跑大量資料。
'''


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
    將人工標示的連續數字 8 上唇下唇頂點畫在同一張 speaker 的 frame 上，顯示該 speaker 嘴巴的最大變動區間。
    '''
    lipCornerPos = CSVtoLIST("./result/lipCorners_upperAndLower/AIZE_R_81139_upperAndLowerLip.csv")
    lipCornerPos = np.array(lipCornerPos,dtype='uint16')
    
    cap = cv2.VideoCapture("./MP4Files/AIZE_R_81139.MP4")
    
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
            cv2.waitKey(0)
            cv2.imwrite("./result/AIZE_maxLipMove.jpg",frame)
            time.sleep(0.02)
        frameID += 1
    
    cv2.destroyAllWindows()


if __name__ == "__main__":
    
    drawLipCorner()