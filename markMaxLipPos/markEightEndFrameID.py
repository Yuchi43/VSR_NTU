#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import csv
import os
import time

import cv2


'''
標出 speaker 念 8 的最後 frameID
'''

def CSVtoDICT(csvFILE):
    '''
    csvDICT scheme >>>
    csvDICT = { 1 : {"order"       : "1",
                     "MPGFILE"     : "MPGFile name",
                    },
                2 : {"order"       : "1",
                     "MPGFILE"     :"MPGFile name",
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
    
    fileName = 'testSheetShort2_8contain.csv'
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
        
    #outputFILE = open("eightEndFrameID.csv", "w")
    #w = csv.writer(outputFILE)
    #w.writerow(eightEndLIST)
    #outputFILE.close()
    #os.system("mv {}_faceOptical_winsize{}.csv ./opticalFlow/faceArea".format(csvDICT[stepLog]["MP4FILE"][:-4],winsize))