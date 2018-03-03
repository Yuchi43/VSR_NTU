#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
import os
import copy

import cv2

def roiSelection(image,fileName):      # 框選 ROI 區域
    global ix,iy,fx,fy,tx,ty,drawing
    img_width, img_height= (len(image[0]),len(image))
    
    cv2.namedWindow('draw face bounding box: {}'.format(fileName),1)
    cv2.resizeWindow('draw face bounding box: {}'.format(fileName), img_width, img_height)
    cv2.setMouseCallback('draw face bounding box: {}'.format(fileName),drawROIregion)   # 以 mouse 拖移來控制 ROI 區域
    
    while True:
        cv2.imshow('draw face bounding box: {}'.format(fileName),image)
        if drawing:                     # mouse 還按著的情況下，reset image, img_width, img_height
            image = copy.deepcopy(img)
            cv2.rectangle(image, (ix,iy), (tx,ty), (0,255,0), 1)
        else:
            pass
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
            
    cv2.destroyAllWindows()

def drawROIregion(event,x,y,flags,param):   # 偵測框選 ROI 的滑鼠事件
    global ix,iy,fx,fy,tx,ty,drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y
        
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img_draw,(ix,iy),(x,y),(0,255,0),1)
        fx,fy = x,y
        
    tx,ty = x,y

def CSVReader(csvFILE):
    '''
    csvDICT scheme >>>
    csvDICT = { 1 : {"order"       : "1",
                     "MPGFILE"     : "path_to_MPGFile",
                     "boundingBoxInitialx"  : "rectangle_leftTop_x_coordinate",
                     "boundingBoxInitialy"  : "rectangle_leftTop_y_coordinate",
                     "boundingBoxFinalx"    : "rectangle_rightDown_x_coordinate",
                     "boundingBoxFinaly"    : "rectangle_rightDown_y_coordinate"
                    },
                2 : {"order"       : "1",
                     "MPGFILE"     :"path_to_MPGFile",
                     "boundingBoxInitialx"  : "rectangle_leftTop_x_coordinate",
                     "boundingBoxInitialy"  : "rectangle_leftTop_y_coordinate",
                     "boundingBoxFinalx"    : "rectangle_rightDown_x_coordinate",
                     "boundingBoxFinaly"    : "rectangle_rightDown_y_coordinate"
                    },
                3...
              }
    '''
    csvDICT = {}
    csvFILE = open(csvFILE, "r")
    seperator = csv.Sniffer().sniff(csvFILE.read(1024), delimiters = ",;\t")
    csvFILE.seek(0)
    csvObj = csv.reader(csvFILE, seperator)
    stepLog = 1
    for i in csvObj:
        csvDICT[stepLog] = {"order"        : i[0],
                            "MP4FILE"      : i[1],
                            "boundingBoxInitialx"  : None,
                            "boundingBoxInitialy"  : None,
                            "boundingBoxFinalx"    : None,
                            "boundingBoxFinaly"    : None
                            }
        stepLog += 1
    return csvDICT

if __name__=='__main__':
    
    try:
        csvDICT = CSVReader("./testSheetShort.csv")
        for key in csvDICT.keys():
            if os.path.exists("./MP4Files/"+csvDICT[key]["MP4FILE"]):
                pass
            else:
                raise SystemExit
        stepLog = 1
    except:
        raise SystemExit
    
    while True:
        if stepLog > len(csvDICT.keys()):
            #寫入檔案
            outputLIST = []
            for i in csvDICT.keys():
                outputLIST.append([csvDICT[i]["MP4FILE"],csvDICT[i]["boundingBoxInitialx"],csvDICT[i]["boundingBoxInitialy"],csvDICT[i]["boundingBoxFinalx"],csvDICT[i]["boundingBoxFinaly"]])
            outputFILE = open("manMade_face_boundingBox.csv", "w")
            w = csv.writer(outputFILE)
            w.writerows(outputLIST)
            outputFILE.close()
            raise SystemExit
        
        cap = cv2.VideoCapture("./MP4Files/"+csvDICT[stepLog]["MP4FILE"])
        cap.set(1,1)       # 取 video 中央的 frame 來畫 bounding box
        ret, img = cap.read()
            
        ix,iy = -1,-1    # 矩形左上角的點
        fx,fy = -1,-1    # 矩形右下角的點
        tx,ty = -1,-1    # mouse 在移動中的點
        drawing = False  # true if mouse is pressed
        img_draw = copy.deepcopy(img)
        roiSelection(img_draw,csvDICT[stepLog]["MP4FILE"])  # 框選 ROI
        
        csvDICT[stepLog]["boundingBoxInitialx"], csvDICT[stepLog]["boundingBoxInitialy"], csvDICT[stepLog]["boundingBoxFinalx"], csvDICT[stepLog]["boundingBoxFinaly"] = ix,iy,fx,fy
        stepLog += 1
        