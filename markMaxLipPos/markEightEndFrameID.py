#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import csv
import os
import time

import cv2


'''
這支程式是輔助工具，能協助人工標出 speaker 念完數字 8 當下的 frameID。
執行後會開啟 opencv 視窗一張一張播放影片的 frame，影片若播放到 speaker 唸完數字 8 的 frame 時，
你可以在視窗內任意一個位置點擊滑鼠左鍵，此時的 frameID 會被紀錄在程式中，點擊 'q' 可以換下一部影片。
40 個影片都標記完後，程式會將全部的 frameID 存在 eightEndFrameID.csv 檔案中。


NOTE: testSheetShort_8contain.csv 原始檔案已遺失，現在使用的檔案只是示範用途。
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

def markUpperPoint(event,x,y,flags,param):   # 標上唇頂點的位置
    global eightEndLIST, frameID
    if event == cv2.EVENT_LBUTTONDOWN:
        print frameID
        eightEndLIST.append(frameID)
        


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
    
    eightEndLIST = []
    
    while stepLog < len(csvDICT.keys()):
        
        cap = cv2.VideoCapture("./MP4Files/"+csvDICT[stepLog]["MP4FILE"])
        
        ret, frame0 = cap.read()
        
        cv2.namedWindow('mark eight end frameID: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]),1)
        cv2.resizeWindow('mark eight end frameID: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]), len(frame0[0]), len(frame0))
        cv2.setMouseCallback('mark eight end frameID: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]),markUpperPoint)
        
        frameID = 1
        while True:
            ret, frame = cap.read()
            if frame == None:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            cv2.imshow('mark eight end frameID: {}'.format(csvDICT[stepLog]["MP4FILE"][:4]), frame)
            
            time.sleep(0.02)
            frameID += 1
        
        cv2.destroyAllWindows()
        stepLog += 1
        
    outputFILE = open("eightEndFrameID.csv", "w")
    w = csv.writer(outputFILE)
    w.writerow(eightEndLIST)
    outputFILE.close()