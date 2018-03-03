#!/usr/bin/env python
# -*- coding:utf-8 -*-


# 影像音節擷取的完整演算法
# 流程：
# 1. 以 adaboost 及 haar features 在影片串流中偵測出一張人臉，固定此人臉位置作為影片中其他 frame 約略的人臉位置。
# 2. 在偵測出人臉的那張 frame 中尋找雙眼的位置。以上半張臉為搜尋區域，利用眼睛顏色為黑色的特質找出合理的眼睛對。
#    找合理眼睛對的方法-> 
#    [step1] 將圖片轉成灰階圖
#    [step2] 動態調整門檻值以二元化圖片。灰階值低於門檻值時設值為 1，反之為0
#            人臉的上半區域主要顏色為皮膚色及黑色，故觀察灰階圖的統計直方圖會發現兩個波峰。
#            設直方圖中兩波峰的中央低點為初始門檻值。
#    [step3] 以數學型態學 closing 方式平滑化圖片
#    [step4] label 圖片，計算 connected component。 連通體面積低於 100 者濾除。
#    [step5] 濾除不合理的眼睛連通體。i.e.,面積過大、高大於寬的連通體不可能是眼睛。
#    [step6] 判斷是否還存在兩個 y 值相近的連通體，
#            若大於一對則取 y 值較大的眼睛對(y值較小的對可能是眉毛)；
#            若無合理眼睛對則將二元化圖片的門檻值調低(變嚴格)，重複 step3-step6。
# 3. 計算密集光流法中參數 winsize 的大小。winsize = （眼睛到臉底部距離）*0.3
# 4. 計算密集光流法的時間範圍：由聲音分析後取得的數組起訖時間內部。
# 5. 計算相鄰 frame 的光流大小。根據光流大小矩陣的統計直方圖濾除整張臉平移的成份。
#    以直方圖濾除平移成份的方法->
#    [step1] 計算直方圖 y 軸的高標
#    [step2] 從直方圖 x 軸的右側往左側觀察，設定碰上的第一個 y 值大於高標的 x 值即為門檻值。
#    [step3] 在光流大小矩陣中，計算高於門檻值的光流值總和，此總和即為該時間下的脣形變動量。
# 6. 標示訊號光流大小值大於區域高標的位置，紀錄兩尖點中最低點的位置，回傳低點位置的時間。


import csv
import os
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt



class SpeechSyllable():
    def __init__(self):
        self.facePosition = None
        self.eyesPosition = None
        self.optWinsize = None
        pass
    
    def getFacePosition(self, rgbImg):
        '''
        input: RGB 影像
        output: 
            *有找到臉時輸出臉在影像中的位置， 即輸出 [faceBbLeftTopX, faceBbLeftTopY, faceBbWidth, faceBbHeight]，
            其中 faceBbLeftTopX(臉框左上角點的 x 值), faceBbLeftTopY(臉框左上角點的 y 值),
            faceBbWidth(臉框的寬度， x 方向), faceBbHeight(臉框的高度， y 方向)
            ---------------------------
            *沒找到臉則輸出 []， 空清單
        '''
        pass
    
    def getEyesPosition(self, faceRgbImg):
        '''
        input: 人臉的 RGB 影像
        output: 
            *有偵測到眼睛時，兩眼的位置，即 [leftEyeX, leftEyeY, rightEyeX, rightEyeY]
            *沒偵測到眼睛則回傳 [], 空清單
        '''
        pass
    
    def getAudioSignalTime(self, fileName):
        '''
        input: 影片的檔案名稱，e.g. xxx.MP4
        output: 由聲音訊號分析取得影片中連續訊號組各自的起訖時間，
                即 [[訊號1起，訊號1訖],[訊號2起，訊號2訖],[訊號3起，訊號3訖],...]，
                舉例說明，呼叫 getAudioSignalTime() 將回傳 [[0.3,1.2],[1.5,2.1],[2.5,3.1],...]
                list 中數值的單位為秒(s)。
        '''
        pass
    
    def getOptWinsize(self):
        '''
        output:the winsize paremeter of the dense optical flow
        '''
        faceBbLeftTopX, faceBbLeftTopY, faceBbWidth, faceBbHeight = self.facePosition
        leftEyeX, leftEyeY, rightEyeX, rightEyeY = self.eyePosition
        self.optWinsize = (faceBbHeight-(leftEyeY+rightEyeY)/2.0)*0.3
        return self.optWinsize
    
    def calcDenseOpt(self, fileName, startFrameID=None, endFrameID=None):
        '''
        input: fileName 影片檔名，
               startFrameID 計算密集光流法的起始 frame 數，預設為 0
               endFrameID 計算密集光流法的結束 frame 數，預設為影片的總長
        output: 回傳兩個 list，即取得影片的光流的 frame 位置 list 及光流的大小值 list，
                即 frameIdList = [frameID1,frameID2,frameID3,...]
                   optMagList = [optMag1,optMag2,optMag3,...]
                也就是說，在 frameID1 時間下計算取得的光流大小值為 optMag1，以此類推。
                
        '''
        cap = cv2.VideoCapture("./MP4Files/"+fileName)
        totalFrame = int(cap.get(7))
        videoFPS = int(round(cap.get(5)))
        
        if startFrame is None:
            startFrame = 0
        if endFrame is None:
            endFrame = totalFrame
        
        # ====> 調整影片為 30fps 再計算光流
        
        
        pass
    
    def findImageSyllable(self, frameIdList, optMagList):
        '''
        input: 呼叫 calcDenseOpt() 後取得的 frameIdList 及 optMagList
        output： 影片的音節 list，即 [0.35,0.4,0.8,...]，list 中數值單位為秒(s)
        '''
        pass
        

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

def main(fileName, syllable):
    try:
        csvDICT = CSVtoDICT("./{}".format(fileName))
        for key in csvDICT.keys():
            if os.path.exists("./MP4Files/"+ csvDICT[key]["MP4FILE"]):
                pass
            else:
                print "{} video file load error!".format(csvDICT[key]["MP4FILE"])
                raise SystemExit
        stepLog = 0
    except:
        print "{} file load error!".format(fileName)
        raise SystemExit
    
    while stepLog <= (len(csvDICT.keys())-1):
        fileName =csvDICT[stepLog]["MP4FILE"]
        cap = cv2.VideoCapture("./MP4Files/"+fileName)
        videoFPS = int(round(cap.get(5)))
        ret, rgbFrame = cap.read()
        facePosition = syllable.getFacePosition(rgbFrame)
        if facePosition:
            faceBbLeftTopX, faceBbLeftTopY, faceBbWidth, faceBbHeight = facePosition
            faceRgbFrame = copy.deepcopy(rgbFrame[faceBbLeftTopY:faceBbLeftTopY+faceBbHeight,faceBbLeftTopX:faceBbWidth])
            eyePosition = syllable.getEyesPosition(faceRgbFrame)
            if eyePosition:
                optWinsize = syllable.getOptWinsize()
                print "winsize===>", optWinsize
                
                audioSignalPeriod = syllable.getAudioSignalTime(fileName)
                imageSyllableLIST = []
                for signalPeriod in audioSignalPeriod:
                    signalBeginFrameID, signalEndFrameID = signalPeriod*videoFPS
                    frameIdList, optMagList = syllable.calcDenseOpt(fileName,signalBeginFrameID,signalEndFrameID)
                    syllableLIST = syllable.findImageSyllable(frameIdList, optMagList)
                    imageSyllableLIST.append(syllableLIST)
                
        # ===== draw matplotlib figure =====





if __name__ == "__main__":
    
    testSheetFile = "testSheetShort.csv"
    syllable = SpeechSyllable()
    main(testSheetFile, syllable)


