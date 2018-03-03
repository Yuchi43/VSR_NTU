#!/usr/bin/env python
# -*- coding:utf-8 -*-


# 校正人臉平移狀況
# Geometric matching 演算法
# 取得人臉鼻子區域的 SIFT 特徵點對(feature pair)，接著疊合其中一對，計算其他特徵點對的疊合程度(RMSD)
# RMSD 愈小則疊合程度愈高，以該疊合點為基準點，此疊合點對的位移即為臉的位移成份。
# 根據基準點的位移校正人臉的位置。
#==================================================================================================

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
import math
from math import factorial
import json
from TextGridParser import TextGridParser

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
from  scipy import ndimage as nd
from skimage.measure import regionprops

frameRateLIST = [8, 12, 20, 24, 30, 40, 60, 120]
outputPath = "./visualProjectResult/"
outputPath4Eye = "./visualProjectResult/eyeDetect/"
outputPath4Opt = "./visualProjectResult/groundTrueAnalysis/slideWinInterval/"
TextGridPath = "./experiment_data/Database (2time-manual)/"

class SpeechSyllable:
    def __init__(self):
        pass
    
    def getEyesPosition(self, faceRgbImg, fileName):
        '''
        以數學型態學 closing 及連通體判斷，找出眼睛對(pair)的位置
        input: 人臉的 RGB 影像
        output: 
            *有偵測到眼睛時，兩眼的位置，即 [leftEyeX, leftEyeY, rightEyeX, rightEyeY]
            *沒偵測到眼睛則回傳 None
        '''
        pureFileName = fileName[:-4]
        self.saveFigPath = outputPath4Eye+pureFileName
        if not os.path.exists(self.saveFigPath):
            os.mkdir(self.saveFigPath)
        #======================================
        #===== find the eye center
        #======================================
        faceGrayImg = cv2.cvtColor(faceRgbImg,cv2.COLOR_BGR2GRAY)
        halfFaceGrayImg = faceGrayImg[:len(faceGrayImg)/2, :]
        firstLocalMin = self.histogramDraw(halfFaceGrayImg, pureFileName)  # 找到二元化門檻值 firstLocalMin
        
        iterCount = 0
        while (iterCount < 21):
            if firstLocalMin < 0:
                print "Can't find eyes with threshold lower than zero after 20 iteration in image."
                return None
            
            th, dst = cv2.threshold(halfFaceGrayImg, firstLocalMin, 255, cv2.THRESH_BINARY_INV)
            
            # morphology
            EllipKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            closingImg = cv2.morphologyEx(dst,cv2.MORPH_CLOSE,kernel=EllipKernel)
            
            # label image
            lab, n = nd.label(closingImg)
            labLIST = lab.ravel().tolist()
            labResultLIST =  ([[x,labLIST.count(x)] for x in set(labLIST)])
            
            for item in labResultLIST:
                if item[1] < 100:
                    np.place(lab,lab == item[0], 0)
            
            #self.colorComponent(lab, "{}/{}_component.png".format(self.saveFigPath, pureFileName))
            
            # 找到每個連通體的特徵值- 質心, bounding box
            componentProps = []   # 紀錄合理的眼睛連通體質心與 bounding box row range
            boundingImg = cv2.cvtColor(closingImg,cv2.COLOR_GRAY2BGR)
            boundingImg1 = cv2.cvtColor(closingImg,cv2.COLOR_GRAY2BGR)
            
            labSetLIST = list(set(lab.ravel().tolist()))
            labNumber = np.arange(0,len(labSetLIST))
            for i, item in enumerate(labSetLIST):
                if item == labNumber[i]:
                    pass
                else:
                    np.place(lab, lab==item, labNumber[i])
            
            regions = regionprops(lab)
            
            for props in regions:
                color = np.random.randint(0,255,(1,3))[0]
                y0, x0 = props.centroid
                minr, minc, maxr, maxc = props.bbox
                #area = props.area
                
                cv2.circle(boundingImg, (int(x0),int(y0)), 4, color,-1)
                cv2.rectangle(boundingImg,(minc,minr),(maxc,maxr),color,2)
                
                if ((maxr-minr) < len(halfFaceGrayImg)/2) and ((maxc-minc) < len(halfFaceGrayImg[0])/2) and ((maxc-minc) > (maxr-minr)):
                    componentProps.append([int(x0), int(y0), minr, minc, maxr, maxc])
                    
            # 儲存每次疊代的結果
            stackImg = np.hstack((faceRgbImg[:len(faceGrayImg)/2, :],boundingImg))
            cv2.imwrite('{}/{}_iter{}.png'.format(self.saveFigPath, pureFileName, iterCount), stackImg)
            
            
            # 判斷質心的合理性
            leftFaceComponent = []
            rightFaceComponent = []
            midFaceCol = len(halfFaceGrayImg[0])/2
            for item in componentProps:
                if item[0] > midFaceCol:
                    rightFaceComponent.append(item)
                else:
                    leftFaceComponent.append(item)

            if (len(leftFaceComponent) == 0) or (len(rightFaceComponent) == 0):      # 臉的左右半邊是否都有 object
                firstLocalMin -= 3
                iterCount += 1
                continue
            else:
                eyePairProb = []
                # 以兩 object Y 軸是否重疊判斷眼睛對
                for (lx0, ly0, lminr, lminc, lmaxr, lmaxc) in leftFaceComponent:
                    leftMask = np.arange(lminr, lmaxr+1)
                    for (rx0, ry0, rminr, rminc, rmaxr, rmaxc) in rightFaceComponent:
                        rightMask = np.arange(rminr, rmaxr+1)
                        mask = np.append(leftMask,rightMask)
                        if len(set(mask)) < len(mask):
                            eyePairProb.append([[lx0, ly0, lminr, lminc, lmaxr, lmaxc],[rx0, ry0, rminr, rminc, rmaxr, rmaxc]])

                eyePair = len(eyePairProb)
                yValue = []
                trueEyePair = None
                if eyePair >= 2:      # 眼睛對大於2的情況，留下 Y 值最大的眼睛對
                    for pair in eyePairProb:
                        yValue.append(pair[0][1]+pair[1][1])
                    trueEyeIndex = yValue.index(max(yValue))
                    trueEyePair = eyePairProb[trueEyeIndex]
                elif eyePair == 1:
                    trueEyePair = eyePairProb[0]
                else:
                    firstLocalMin -= 3
                    iterCount += 1
                    continue

                if trueEyePair:
                    color = np.random.randint(0,255,(1,3))[0]
                    for eyeObj in trueEyePair:
                        cv2.circle(boundingImg1, (eyeObj[0],eyeObj[1]), 4, color,-1)
                        cv2.rectangle(boundingImg1,(eyeObj[3],eyeObj[2]),(eyeObj[5],eyeObj[4]),color,2)
                    
                    stackImg1 = np.hstack((faceRgbImg[:len(faceGrayImg)/2, :],boundingImg1))
                    cv2.imwrite('{}/{}_eye_component.png'.format(self.saveFigPath,pureFileName),stackImg1)
                    
                    return [trueEyePair[0][0], trueEyePair[0][1], trueEyePair[1][0], trueEyePair[1][1]]
                else:
                    firstLocalMin -= 3
                    iterCount += 1
                    
        if iterCount == 21:
            print "{} can't find eye after 20 iteration".format(pureFileName)
            return None
    
    def histogramDraw(self, matrix, fileName):   # 畫出 histogram
        maxvalue = matrix.max()
        #---histogram
        bins = np.arange(maxvalue + 1)
        hist,bins = np.histogram(matrix,bins)
        histSmooth = self.savitzky_golay(hist, 37, 1)
    
        histSlopeLIST = []
        for i, item in enumerate(histSmooth):
            if i < len(histSmooth)-1:
                histSlopeLIST.append(histSmooth[i+1]-histSmooth[i])
            if i == len(histSmooth)-1:
                histSlopeLIST.append(0)
        histSlopeLIST = np.array(histSlopeLIST)
        histSlopeLIST = histSlopeLIST > 0
        
        stateChangeIndex = []
        for i,item in enumerate(histSlopeLIST):
            if i == 0:
                state = item
                stateChangeIndex.append(i)
            else:
                if item == state:
                    pass
                else:
                    state = item
                    stateChangeIndex.append(i)
    
        if len(stateChangeIndex) >= 3:
            firstLocalMin = stateChangeIndex[2]
        else:
            firstLocalMin = 50
        
        width = bins[1] - bins[0]
        center = (bins[:-1]+bins[1:])/float(2)
        
        fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True)
        ax0.bar(center, hist, width=width)
        ax0.set_title("origin")
        ax1.bar(center, histSmooth, width=width)
        ax1.set_title("smoothing")
        ax1.axvline(x=firstLocalMin, linewidth=2, color='r')
        
        fig.savefig('{}/{}_halfFaceHist.png'.format(self.saveFigPath,fileName),bbox_inches='tight')
        plt.clf()
        
        return firstLocalMin
    
    def savitzky_golay(self, y, window_size, order, deriv=0, rate=1):
        r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
        The Savitzky-Golay filter removes high frequency noise from data.
        It has the advantage of preserving the original shape and
        features of the signal better than other types of filtering
        approaches, such as moving averages techniques.
        Parameters
        ----------
        y : array_like, shape (N,)
        the values of the time history of the signal.
        window_size : int
        the length of the window. Must be an odd integer number.
        order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
        deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
        Returns
        -------
        ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
        Notes
        -----
        The Savitzky-Golay is a type of low-pass filter, particularly
        suited for smoothing noisy data. The main idea behind this
        approach is to make for each point a least-square fit with a
        polynomial of high order over a odd-sized window centered at
        the point.
        Examples
        --------
        t = np.linspace(-4, 4, 500)
        y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
        ysg = savitzky_golay(y, window_size=31, order=4)
        import matplotlib.pyplot as plt
        plt.plot(t, y, label='Noisy signal')
        plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
        plt.plot(t, ysg, 'r', label='Filtered signal')
        plt.legend()
        plt.show()
        References
        ----------
        [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
        Data by Simplified Least Squares Procedures. Analytical
        Chemistry, 1964, 36 (8), pp 1627-1639.
        [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
        Cambridge University Press ISBN-13: 9780521880688
        """
        try:
            window_size = np.abs(np.int(window_size))
            order = np.abs(np.int(order))
        except ValueError, msg:
            raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")
        order_range = range(order+1)
        half_window = (window_size -1) // 2
        # precompute coefficients
        b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
        m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
        # pad the signal at the extremes with
        # values taken from the signal itself
        firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
        lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
        y = np.concatenate((firstvals, y, lastvals))
        return np.convolve( m[::-1], y, mode='valid')
    
    def colorComponent(self, labImg, saveFigName):
        # 以顏色區分連通體
        colorImg = np.array(np.tile(labImg[:,:,np.newaxis],3),dtype="uint8")
        for labelNum in set(labImg.ravel().tolist()):
            if labelNum:
                color = np.random.randint(0,255,(1,3))[0]
                mask = np.all(colorImg == [labelNum]*3, axis=2)
                colorImg[mask] = color
        cv2.imwrite(saveFigName,colorImg)
    
    def getSyllableGroundTruth(self, videoName):
        '''
        取得 Praat 人工切的影片音節位置，output 格式為 [x1, x2, x3, x4, ...]，x 為音節位置的時間 
        '''
        pureVideoName = videoName[:-4]
        speakerName = videoName[:4]
        kind = videoName[5]
        if kind == 'R':
            kind = 'reading'
        else:
            kind = 'speaking'
        
        tg = TextGridParser("{}{}/{}_{}/{}.TextGrid".format(TextGridPath, speakerName, speakerName, kind, pureVideoName))
        signalInterval = tg.TextGridDICT["item_1"]["intervals"]
        audioGroundTruth = []
        audioGroundTruthContent = []
        for i, syllable in enumerate(signalInterval):
            if syllable[-1]:
                if syllable[1] not in audioGroundTruth:
                    audioGroundTruth.append(syllable[1])
                audioGroundTruth.append(syllable[2])
                audioGroundTruthContent.append(syllable[3])
        
        return audioGroundTruth, audioGroundTruthContent
    
    def getOptWinsize(self, facePosition, eyePosition):
        '''
        output:the winsize paremeter of the dense optical flow
        '''
        faceBbLeftTopX, faceBbLeftTopY, faceBbRightBottomX, faceBbRightBottomY = facePosition
        faceBbWidth = faceBbRightBottomX - faceBbLeftTopX
        faceBbHeight = faceBbRightBottomY - faceBbLeftTopY
        
        leftEyeX, leftEyeY, rightEyeX, rightEyeY = eyePosition
        optWinsize = int((faceBbHeight-(leftEyeY+rightEyeY)/2.0)*0.3)
        return optWinsize
    
    def drawOptFigure(self, videoName, optMagLIST1, optMagLIST2, optPeakPos, optMinPos, audioGroundTruth, frameRate):  # , optMagLIST3, optMagLIST4
        '''
        畫出光流大小折線圖(固定人臉框、sift 調整人臉框)、波形能量圖及人工標變動時間圖。
        input: videoName - 應用光流法的影片名稱
               optMagLIST1 - 固定人臉框的光流大小，資料型態為 list。
               optMagLIST2 - sift 調整人臉框後的光流大小，資料型態為 list。
               optMagLIST3 - sift + area matching 調整人臉框後的光流大小，資料型態為 list。
        output: matplotlib figure。
        '''
        pureVideoName = videoName[:-4]
        speakerName = videoName[:4]
        kind = videoName[5]
        if kind == 'R':
            kind1 = 'reading'
            kind2 = 'Reading'
        else:
            kind1 = 'speaking'
            kind2 = 'Speaking'
        
        #=====取得影像音節
        obj = GeometricMatching()
        imgSyllableLIST = obj.csvToList("./experiment_data/{}/{}_{}/{}_image_syllable_man.csv".format(speakerName,speakerName,kind2,pureVideoName))[0]
        imgSyllableLIST = np.array(imgSyllableLIST,dtype='uint16')
        imgSyllableLIST = imgSyllableLIST/120.0
        #=====取得聲音波形
        wavFile = open("./experiment_data/Database (wavNormalized)/{}/{}_{}/{}.json".format(speakerName,speakerName,kind1,pureVideoName), "r")
        wavText = wavFile.read()
        wavLIST = json.loads(wavText)
        #=====取得聲音音節
        #wavPath = "./experiment_data/audioSyllable/{}.json".format(pureVideoName)
        #wavFileFlag = False
        #if os.path.exists(wavPath):
            #wavFileFlag = True
            #syllableFile = open(wavPath, "r")
            #syllableText = syllableFile.read()
            #syllableLIST = json.loads(syllableText)
            #audioSyllableOdd = []
            #audioSyllableEven = []
            #for j, syllable in enumerate(syllableLIST):
                #if j % 2 == 1:
                    #for subItem in filter(self.filterFunc,syllable):
                        #audioSyllableOdd.append(subItem)
                #else:
                    #for subItem in filter(self.filterFunc,syllable):
                        #audioSyllableEven.append(subItem)
        
        axisX_opt = np.arange(0,50,1.0/frameRate)
        axisX_opt = axisX_opt[1:len(optMagLIST1)+1]
        
        axisX_wav = np.arange(0,50,1.0/(44100))
        axisX_wav = axisX_wav[:len(wavLIST)]
        
        fig = plt.figure()
        DPI = fig.get_dpi()
        fig.set_size_inches(3113/float(DPI),434*2.5/float(DPI))
        matplotlib.rcParams.update({'font.size': 16})
        #fig.set_size_inches(21.5,6.5, forward = True)
        gs2 = gridspec.GridSpec(5, 1)
        ax1 = fig.add_subplot(gs2[0,0])
        ax2 = fig.add_subplot(gs2[1,0], sharex=ax1)
        ax7 = fig.add_subplot(gs2[2,0], sharex=ax1)
        ax5 = fig.add_subplot(gs2[3,0], sharex=ax1)
        ax6 = fig.add_subplot(gs2[4,0], sharex=ax1)
        gs2.update(hspace=0.5)
        
        # ax1 未經處理的光流大小總和折線圖
        plot1, = ax1.plot(axisX_opt,optMagLIST1,color = 'mediumblue',linewidth=2)
        ax1.set_ylim(0.0,max(optMagLIST1))
        ax1.set_yticks(np.arange(0,1.5,0.5))
        ax1.set_title('{}  halfFace optical flow magnitude sum (pure, {}fps)'.format(pureVideoName, frameRate))
        #ax1.legend([plot1,],('flow magnitude',), numpoints = 3)
        
        # ax2 以 sift 調整過的光流大小總和折線圖
        plot2, = ax2.plot(axisX_opt,optMagLIST2,color = 'mediumblue',linewidth=2)
        for item in optPeakPos:
            ax2.axvline(x=item, linewidth=2, color='gray')
        ax2.set_ylim(0.0,max(optMagLIST2))
        ax2.set_yticks(np.arange(0,1.5,0.5))
        ax2.set_title('{}  halfFace optical flow magnitude sum (sift_square, {}fps)'.format(pureVideoName, frameRate))
        #ax2.legend([plot2,],('flow magnitude',), numpoints = 3)
        
        # ax3 以 sift + area matching 調整過的光流大小總和折線圖
        #plot3, = ax3.plot(axisX_opt,optMagLIST3,color = 'mediumblue',linewidth=2)
        #ax3.set_ylim(0.0,max(optMagLIST3))
        #ax3.set_title('{}  halfFace optical flow magnitude sum (sift_RMSD)'.format(pureVideoName))
        #ax3.legend([plot2,],('flow magnitude',), numpoints = 3)
        
        # ax4 以 sift + area matching + angle filter 調整過的光流大小總和折線圖
        #plot4, = ax4.plot(axisX_opt,optMagLIST4,color = 'mediumblue',linewidth=2)
        #ax4.set_ylim(0.0,max(optMagLIST4))
        #ax4.set_title('{}  halfFace optical flow magnitude sum (sift_RMSD_square)'.format(pureVideoName))
        #ax4.legend([plot2,],('flow magnitude',), numpoints = 3)
        
        # ax7 波型能量圖(標示變動量最低點)
        plot7, = ax7.plot(axisX_wav,wavLIST,color = 'black',linewidth=0.5)
        for item in optMinPos:
            ax7.axvline(x=item, linewidth=2, color='forestgreen')
        ax7.set_ylim(min(wavLIST),max(wavLIST))
        ax7.set_yticks(np.arange(-1,1.5,0.5))
        ax7.set_title('{} wav energy(mark minimum position)'.format(pureVideoName))
        #ax7.legend([plot7,],('wav energy',), numpoints = 3)
        
        # ax5 波型能量圖(ground truth)
        plot5, = ax5.plot(axisX_wav,wavLIST,color = 'black',linewidth=0.5)
        for item in audioGroundTruth:
            ax5.axvline(x=item, linewidth=2, color='r')
        #if wavFileFlag:
            #for item in audioSyllableEven:
                #ax5.axvline(x=item, linewidth=2, color='r')
            #for item in audioSyllableOdd:
                #ax5.axvline(x=item, linewidth=2, color='orange')
        ax5.set_ylim(min(wavLIST),max(wavLIST))
        ax5.set_yticks(np.arange(-1,1.5,0.5))
        ax5.set_title('{} wav energy (ground truth)'.format(pureVideoName))
        #ax5.legend([plot5,],('wav energy',), numpoints = 3)
        
        # ax6 影像上人工切的音節位置
        for item in imgSyllableLIST:
            ax6.axvline(x=item, linewidth=2, color='gray')
        ax6.set_ylim(0.0,1)
        ax6.set_yticks(np.arange(0,2,1))
        ax6.set_title("{} image syllable by man".format(pureVideoName))
        #------------------
        plt.xlim(0,max(axisX_opt))
        plt.xticks(np.arange(0,max(axisX_opt),0.5))
        #plt.yticks(np.arange(0,1.5,0.5))
        plt.xlabel('Time(sec.)')
        
        #plt.show()
        fig.savefig('{}optFig/{}_{}fps_squareOpt.png'.format(outputPath4Opt, pureVideoName, frameRate),bbox_inches='tight')
        plt.clf()
        plt.close()
        
    def angleFilterAnalysis(self, siftAngMatrix):
        bins = np.arange(360 + 1)
        siftAngHist,bins = np.histogram(siftAngMatrix,bins)
        siftIndex = np.argmax(siftAngHist)
        leftSumIndex = siftIndex - 5
        rightSumIndex = siftIndex + 5
        angFilter = np.array(np.zeros_like(siftAngMatrix), dtype='bool')
    
        if (leftSumIndex >= 0) and (rightSumIndex <= 360):
            siftAngHistSumLIST = siftAngHist[leftSumIndex:rightSumIndex+1]
            siftAngMaxCount = np.sum(siftAngHist[leftSumIndex:rightSumIndex+1])
            if siftAngMaxCount >= (len(siftAngMatrix)*len(siftAngMatrix[0]))/2:
                angFilter = (siftAngMatrix > bins[leftSumIndex]) * (siftAngMatrix < bins[rightSumIndex]+1)
            
        elif (leftSumIndex < 0):
            overFlowHist = siftAngHist[leftSumIndex:]
            siftAngHistSumLIST = np.append(siftAngHist[:rightSumIndex+1],overFlowHist)
            siftAngMaxCount = np.sum(np.append(siftAngHist[:rightSumIndex+1],overFlowHist))
            if siftAngMaxCount >= (len(siftAngMatrix)*len(siftAngMatrix[0]))/2:
                angFilter = (siftAngMatrix < bins[rightSumIndex]+1) + (siftAngMatrix > bins[leftSumIndex])
            
        elif (rightSumIndex > 360):
            overFlowHist = siftAngHist[0:rightSumIndex-359]
            siftAngHistSumLIST = np.append(siftAngHist[leftSumIndex:],overFlowHist)
            siftAngMaxCount = np.sum(np.append(siftAngHist[leftSumIndex:],overFlowHist))
            if siftAngMaxCount >= (len(siftAngMatrix)*len(siftAngMatrix[0]))/2:
                angFilter = (siftAngMatrix < bins[rightSumIndex-360]) + (siftAngMatrix > bins[leftSumIndex])
        
        return angFilter
        
    def findImageSyllable(self, videoName, optMagList, frameRate):
        '''
        input: videoName 影片檔名稱；optMagList 光流大小 list。
        output： optPeakPos -> 計算每個數組區間的高標值，儘保留該區間中高於高標的光流大小峰值位置。格式為 [[peak00, peak01, peak02], [peak10, peak11, peak12], [peak20, peak21]] 
                               ([[區間0], [區間1], [區間2]]), peak 單位為秒(s)。
                 optMinPos -> optPeakPos 中對應的最低點。格式為 [[min00, min01], [min10, min11], [min20]], min 單位為秒(s)。
                 audioInterval -> Praat 取得的數組起訖時間，格式為 [[startInterval0, endInterval0], [startInterval1, endInterval1], [startInterval2, endInterval2]]
        '''
        pureVideoName = videoName[:-4]
        speakerName = videoName[:4]
        kind = videoName[5]
        if kind == 'R':
            kind = 'reading'
        else:
            kind = 'speaking'
        
        peakPosIndexLIST = []   # 紀錄尖點位置在光流大小折線圖中的索引值
        minPosIndexLIST = []   # 紀錄兩尖點中的最低點在光流大小折線圖中的索引值
    
        totalSignalTime = len(optMagList)/float(frameRate)
        optMagTimeList = np.arange(0, 50, 1.0/frameRate)
        optMagTimeList = optMagTimeList[1:len(optMagList)+1]
        
        winStartTimeLIST = np.arange(0, totalSignalTime, 1.0/3)
        for winStartTime in winStartTimeLIST:
            optMagTimeSubList = filter(lambda x: (x>winStartTime), optMagTimeList)
            startIndex = np.where(optMagTimeList == optMagTimeSubList[0])[0][0]
            if (startIndex+frameRate) >= len(optMagList):
                break
            subOptMagList = optMagList[startIndex:startIndex+frameRate]
            peakPosTemp = []
            for k, item in enumerate(subOptMagList):
                if (k == 0) or (k == len(subOptMagList)-1):
                    pass
                else:
                    if (item > subOptMagList[k-1]) and (item > subOptMagList[k+1]):
                        peakPosTemp.append([k, subOptMagList[k]])
            peakPosTemp.sort(key=lambda x: x[1], reverse=True)
            if peakPosTemp:
                for peak in np.array(peakPosTemp)[:,0][:3]:
                    if (peak+startIndex) not in peakPosIndexLIST:
                        peakPosIndexLIST.append(int(peak+startIndex))
                
        peakPosIndexLIST.sort()
        for i, peakIndex in enumerate(peakPosIndexLIST):
            if (i == len(peakPosIndexLIST)-1):
                pass
            else:
                lineChartPatch = optMagList[peakIndex:peakPosIndexLIST[i+1]]
                patchMinIndex = peakIndex + np.argmin(lineChartPatch)
                minPosIndexLIST.append(patchMinIndex)
        
        #minPosIndexLIST.sort()
        peakPos = [(x+1)/float(frameRate) for x in peakPosIndexLIST]
        minPos = [(x+1)/float(frameRate) for x in minPosIndexLIST]
        
        # peakPos 寫檔
        outputFILE = open("{}_{}fps_peakPos.csv".format(pureVideoName, frameRate), "w")
        w = csv.writer(outputFILE)
        w.writerow(peakPos)
        outputFILE.close()
        os.system("mv {}_{}fps_peakPos.csv {}imgSyllableCSV".format(pureVideoName, frameRate, outputPath4Opt))
        # minPos 寫檔
        outputFILE = open("{}_{}fps_minPos.csv".format(pureVideoName, frameRate), "w")
        w = csv.writer(outputFILE)
        w.writerow(minPos)
        outputFILE.close()
        os.system("mv {}_{}fps_minPos.csv {}imgSyllableCSV".format(pureVideoName, frameRate, outputPath4Opt))
        
        return peakPos, minPos


class GeometricMatching:
    def __init__(self):
        pass
    
    def getFacePosition(self, isFirstFrame = False, videoName = None, baseRgbFrame = None, baseFacePosition = None, targetRgbFrame = None, showSiftResult = False, matchMethod = "RMSD"):
        '''
        以人工框的一張 frame 人臉為基準，藉由 geometric matching 方法取得當前影片的第 1 張人臉位置。
        input: isFirstFrame - (boolean) 預設為 False。
                              if True:當前欲偵測人臉的 frame 為影片的第 1 張 frame，需給定 videoName 及 targetRgbFrame 參數。
                              if False:當前欲偵測人臉的 frame 非影片的第 1 張 frame，需給定 baseRgbFrame、baseFacePosition 及 targetRgbFrame 參數。
               videoName-欲處理的影片名稱
               baseRgbFrame - 前一張 frame。(RGB 圖片)
               baseFacePosition - 前一張 frame 中的人臉座標位置，即 [faceBbLeftTopX, faceBbLeftTopY, faceBbRightBottomX, faceBbRightBottomY]。
               targetRgbFrame - 當下欲偵測人臉的 frame。(RGB 圖片)
               
        output: 人臉在 frame 中的 Bounding box，即 [faceBbLeftTopX, faceBbLeftTopY, faceBbRightBottomX, faceBbRightBottomY]，
            其中 faceBbLeftTopX(臉框左上角點的 x 值), faceBbLeftTopY(臉框左上角點的 y 值),
            faceBbRightBottomX(臉框右下角點的 x 值), faceBbRightBottomY(臉框右下角點的 y 值),
        '''
        if isFirstFrame:
            facePosLIST = self.csvToList("./manMade_face_boundingBox.csv")
            facePosDict = {}
            for item in facePosLIST:
                facePosDict['{}'.format(item[0][:4])] = {"manMadeVideoName": item[0],
                                                         "facePosition":np.array(item[1:],dtype='uint16'),
                                                         }
            
            speakerName = videoName[:4]
            cap = cv2.VideoCapture("./MP4Files/"+facePosDict[speakerName]["manMadeVideoName"])
            ret, baseRgbFrame = cap.read()
            cap.release()
            
            self.facePositionLIST = facePosDict[speakerName]["facePosition"]
            baseFaceIx, baseFaceIy, baseFaceFx, baseFaceFy = self.facePositionLIST
        else:
            self.facePositionLIST = baseFacePosition
            baseFaceIx, baseFaceIy, baseFaceFx, baseFaceFy = self.facePositionLIST
            
        siftPairs = self.getSiftFeature(baseRgbFrame, targetRgbFrame)
        restrictSiftPairs = self.faceAreaSIFTpoint(siftPairs)
        if showSiftResult:
            #self.saveSiftResult(siftPairs, baseRgbFrame, targetRgbFrame, "originSIFT",True)
            self.saveSiftResult(restrictSiftPairs, baseRgbFrame, targetRgbFrame, "faceAreaSIFT",True)
        if len(restrictSiftPairs) <= 1:
            if isFirstFrame:
                return self.facePositionLIST
            else:
                return None
        
        siftPairForMap = self.pointsMatching(restrictSiftPairs, baseRgbFrame, targetRgbFrame, matchMethod)
        faceShiftX, faceShiftY = self.calcFaceShift(siftPairForMap)
        facePosition = [baseFaceIx+faceShiftX, baseFaceIy+faceShiftY, baseFaceFx+faceShiftX, baseFaceFy+faceShiftY]
        self.facePositionLIST = facePosition
        return facePosition
    
    def getSiftFeature(self, baseRgbImage, targetRgbImage):
        '''
        input: baseImage -> 基準影像(RGB)。
               targetImage -> 欲比對的影像(RGB)。
        output: 穩定的 SIFT 特徵點對，格式為 [[x1,y1,x1',y1'],[x2,y2,x2',y2'],[x3,y3,x3',y3'],...]
                [x1,y1,x1',y1'] 中 [x1,y1] 是基準影像的特徵點 xy 座標；[x1',y1'] 是欲比對影像中的特徵點 xy 座標。
        '''
        baseGrayImage = cv2.cvtColor(baseRgbImage,cv2.COLOR_BGR2GRAY)
        targetGrayImage = cv2.cvtColor(targetRgbImage,cv2.COLOR_BGR2GRAY)
        
        #========== SIFT MAPPING ==========
        detector = cv2.SIFT()
        FLANN_INDEX_KDTREE = 1
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE,trees = 5)
        matcher = cv2.FlannBasedMatcher(flann_params, {})
        kp1, desc1 = detector.detectAndCompute(baseGrayImage, None)
        kp2, desc2 = detector.detectAndCompute(targetGrayImage, None)
    
        #----matching
        raw_matches = matcher.knnMatch(desc1, desc2, k = 2)
        p1, p2, kp_pairs = self.filter_matches(kp1, kp2, raw_matches)
        #print "p1===>",p1[0]
        #print "p2===>",p2[0]
        #print "kp_pairs===>",kp_pairs[0][0].pt,kp_pairs[0][1].pt
        if len(p1) >= 4:
            H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)
            #print '%d / %d  inliers/matched' % (np.sum(status), len(status))
        else:
            H, status = None, None
            #print '%d matches found, not enough for homography estimation' % len(p1)
    
        robustFeatureSet = []
        for (Point1, Point2), inlier in zip(kp_pairs, status):
            if inlier:
                robustFeatureSet.append([int(round(Point1.pt[0])),int(round(Point1.pt[1])), int(round(Point2.pt[0])), int(round(Point2.pt[1]))])
            else:
                pass
           
        return robustFeatureSet
    
    def filter_matches(self, kp1, kp2, matches, ratio = 0.75):
        mkp1, mkp2 = [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs
    
    def faceAreaSIFTpoint(self, siftPairs):
        '''
        依據人臉的 bounding box 定位出鼻子的位置，儘保留落於鼻子區域內部的 sift 特徵點對。
        input: siftPairs-> 從 getSiftFeature() 取得之 sift 特徵點對
        output: 鼻子區域內部的特徵點對。格式為 [[x1,y1,x1',y1'],[x2,y2,x2',y2'],[x3,y3,x3',y3'],...]
        '''
        baseFaceIx, baseFaceIy, baseFaceFx, baseFaceFy = self.facePositionLIST
        #topNose = baseFaceIy + int(round((baseFaceFy-baseFaceIy)/3.0))
        #bottomNose = baseFaceIy + int(round(2*(baseFaceFy-baseFaceIy)/3.0))
        #leftNose = baseFaceIx + int(round((baseFaceFx-baseFaceIx)/3.0))
        #rightNose = baseFaceIx + int(round(2*(baseFaceFx-baseFaceIx)/3.0))
        
        restrictSiftPairs = []
        for (x1, y1, x2, y2) in siftPairs:
            #if (x1 > leftNose) and (x1 < rightNose) and (y1 > topNose) and (y1 < bottomNose):     # feature point contain in nose region
            if (x1 > baseFaceIx) and (x1 < baseFaceFx) and (y1 > baseFaceIy) and (y1 < baseFaceFy):
                restrictSiftPairs.append([x1, y1, x2, y2])
        #print "restrictPairs===>",len(restrictSiftPairs)
        return restrictSiftPairs
    
    def saveSiftResult(self, siftPairs, baseRgbImage, targetRgbImage, saveFigName, showFigure = False):
        '''
        顯示 SIFT 配對結果。
        '''
        baseGrayImage = cv2.cvtColor(baseRgbImage,cv2.COLOR_BGR2GRAY)
        targetGrayImage = cv2.cvtColor(targetRgbImage,cv2.COLOR_BGR2GRAY)
        
        h1, w1 = baseGrayImage.shape[:2]
        h2, w2 = targetGrayImage.shape[:2]
        
        vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
        vis[:h1, :w1] = baseGrayImage
        vis[:h2, w1:w1+w2] = targetGrayImage
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        
        col = [0,255,0]
        for (x1, y1, x2, y2) in siftPairs:
            cv2.circle(vis, (x1, y1), 2, col, -1)
            cv2.circle(vis, (x2+w1, y2), 2, col, -1)
            cv2.line(vis, (x1, y1), (x2+w1, y2), col, 1)
        
        if showFigure:
            cv2.imshow("SIFT Mappint Result", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        cv2.imwrite(saveFigName+".jpg", vis)
    
    def pointsMatching(self, siftPairs, baseRgbFrame, targetRgbFrame, matchMethod):
        '''
        疊合其中一組特徵點對，計算其餘特徵點對的疊合程度。每個特徵點對都會進行完整疊合。
        input: sift 的特徵點對，格式為 [[x1,y1,x1',y1'],[x2,y2,x2',y2'],[x3,y3,x3',y3'],...]
        output: 疊合程度最高時的疊合基準點。格式為 [x,y,x',y']。
        '''
        if matchMethod == "superimpose":
            baseFaceIx, baseFaceIy, baseFaceFx, baseFaceFy = self.facePositionLIST
            baseRgbFace = baseRgbFrame[baseFaceIy:baseFaceFy+1, baseFaceIx:baseFaceFx+1]
            baseGrayFace = cv2.cvtColor(baseRgbFace, cv2.COLOR_BGR2GRAY)
            baseGrayHalfFace = baseGrayFace[:len(baseGrayFace)/2]
            baseGrayHalfFace = np.array(baseGrayHalfFace, dtype="int16")
            
            targetGrayFrame = cv2.cvtColor(targetRgbFrame, cv2.COLOR_BGR2GRAY)
            targetGrayFrame = np.array(targetGrayFrame, dtype="int16")
            
            minDiffPair = 0
            minDiffValue = 99999999999999
            for i, item in enumerate(siftPairs):
                ix, iy ,fx, fy = item
                xShift = fx - ix
                yShift = fy - iy
                targetGrayFace = targetGrayFrame[baseFaceIy+yShift:baseFaceFy+yShift+1, baseFaceIx+xShift:baseFaceFx+xShift+1]
                targetGrayHalfFace = targetGrayFace[:len(targetGrayFace)/2]
                imgGrayDiffSum = np.square(targetGrayHalfFace - baseGrayHalfFace).sum()
                
                if imgGrayDiffSum < minDiffValue:
                    minDiffValue = imgGrayDiffSum
                    minDiffPair = i
            return siftPairs[minDiffPair]
        
        if matchMethod == "RMSD":
        # === 以 RMSD 求得最佳疊合的 sift pair ===
            maxMatching = 0  # the index of the max matching point in sift pairs
            minRMSD = 1000  # initial RMSD
            for i, item in enumerate(siftPairs):
                ix, iy ,fx, fy = item
                xShift = fx - ix
                yShift = fy - iy
                adjustSiftPairs = copy.deepcopy(siftPairs)
                adjustSiftPairs.remove(item)
                
                pointsDistance = 0
                for j, item2 in enumerate(adjustSiftPairs):
                    ix2, iy2, fx2, fy2 = item2
                    fx2 = fx2 - xShift
                    fy2 = fy2 - yShift
                    pointsDistance = pointsDistance + (math.pow((fx2-ix2),2) + math.pow((fy2-iy2),2))
                    
                rmsd = math.sqrt(float(pointsDistance)/len(adjustSiftPairs))
                if rmsd < minRMSD:
                    minRMSD = rmsd
                    maxMatching = i
            return siftPairs[maxMatching]
    
    def calcFaceShift(self,siftPairForMap):
        '''
        input: FlagSiftPair -> pointMatching() 中取得的疊合基準點。
        output: 疊合基準點的位移，即 [xShift, yShift]
        '''
        ix, iy ,fx, fy = siftPairForMap
        faceShiftX = fx - ix
        faceShiftY = fy - iy
        return [faceShiftX, faceShiftY]
    
    def areaMatching(self, prvsRgbFace, currentRgbFace):
        
        # prev face area
        prevHsvFace = cv2.cvtColor(prvsRgbFace, cv2.COLOR_BGR2HSV)
        (prev_H, prev_S, prev_V) = cv2.split(prevHsvFace)
        mask1 = prev_H<30
        mask2 = prev_V>150
        prevMask = mask1 * mask2
        prevMaskFace = np.array(prevMask,dtype='uint8')
        prevMaskFace = cv2.medianBlur(prevMaskFace, 11)
        prevMaskFaceBool = np.array(prevMaskFace, dtype='bool')
        
        # current face area
        currentHsvFace = cv2.cvtColor(currentRgbFace, cv2.COLOR_BGR2HSV)
        (current_H, current_S, current_V) = cv2.split(currentHsvFace)
        mask1 = current_H<30
        mask2 = current_V>150
        currentMask = mask1 * mask2
        currentMaskFace = np.array(currentMask,dtype='uint8')
        currentMaskFace = cv2.medianBlur(currentMaskFace, 11)
        currentMaskFaceBool = np.array(currentMaskFace, dtype='bool')
        
        # filter inner variation
        twoFaceAND = prevMaskFaceBool * currentMaskFaceBool   # 兩張臉面積有重疊的位置
        firstTruePos1, upDownMask = self.firstTrueMask(twoFaceAND, dirt = "top")
        firstTruePos2, downUpMask = self.firstTrueMask(twoFaceAND, dirt = "bottom")
        firstTruePos3, leftRightMask = self.firstTrueMask(twoFaceAND, dirt = "left")
        firstTruePos4, rightLeftMask = self.firstTrueMask(twoFaceAND, dirt = "right")
        inerMask = upDownMask * downUpMask * leftRightMask * rightLeftMask
        
        prevMaskFaceBool = prevMaskFaceBool + inerMask
        currentMaskFaceBool = currentMaskFaceBool + inerMask
        
        loopCount = 0
        xShift = 0
        yShift = 0
        xShiftAll = 0
        yShiftAll = 0
        prevMatchingRate = 0
        while (loopCount < 20):
            # === calculate matching rate ===
            twoFaceAND = prevMaskFaceBool * currentMaskFaceBool   # 兩張臉面積有重疊的位置
            twoFaceOR = prevMaskFaceBool + currentMaskFaceBool    # 兩張臉的面積總和
            AndArea = np.count_nonzero(twoFaceAND)
            OrArea = np.count_nonzero(twoFaceOR)
            matchingRate = AndArea/float(OrArea)
            
            if matchingRate < prevMatchingRate:
                break
            
                # === 以 inerMask 濾除臉內部的變動 ===
            prevMaskFaceInt = np.array(prevMaskFaceBool,dtype='int8')
            currentMaskFaceInt = np.array(currentMaskFaceBool,dtype='int8')
            shiftMask = currentMaskFaceInt - prevMaskFaceInt
            pMask = shiftMask == 1      # 皮膚區域有多的部份
            
                # === show 變動區域 ===
            matchingShow = np.zeros_like(prvsRgbFace)
            matchingShow[pMask] = [0,255,0]
            #cv2.imshow("test", matchingShow)
            #cv2.waitKey(0)
            
            xShiftAll = xShiftAll + xShift
            yShiftAll = yShiftAll + yShift
            
            if matchingRate > 0.998:
                break
            
            # === 依據臉位移的方向，調整臉框的位置 ===
            currentMaskFaceCanvas = np.zeros_like(currentMaskFace)
            
            p0Count = np.count_nonzero(pMask[:len(pMask)/2, :len(pMask[0])/2])
            p1Count = np.count_nonzero(pMask[:len(pMask)/2, len(pMask[0])/2:])
            p2Count = np.count_nonzero(pMask[len(pMask)/2:, len(pMask[0])/2:])
            p3Count = np.count_nonzero(pMask[len(pMask)/2:, :len(pMask[0])/2])
            
            pCountLIST = [p0Count, p1Count, p2Count, p3Count]
            sortPCount = sorted(pCountLIST, reverse=True)
            firstValue = sortPCount[0]
            secondValue = sortPCount[1]
            firstIndex = pCountLIST.index(firstValue)
            secondIndex = pCountLIST.index(secondValue)
            
            # === 判斷位移 ===
            if (firstIndex == 0 and secondIndex == 1) or (firstIndex == 1 and secondIndex == 0):
                yShift = -1
            elif (firstIndex == 2 and secondIndex == 3) or (firstIndex == 3 and secondIndex == 2):
                yShift = 1
            elif (firstIndex == 1 and secondIndex == 2) or (firstIndex == 2 and secondIndex == 1):
                xShift = 1
            elif (firstIndex == 0 and secondIndex == 3) or (firstIndex == 3 and secondIndex == 0):
                xShift = -1
            elif (firstIndex == 0 and secondIndex == 2) or (firstIndex == 2 and secondIndex == 0):
                if firstIndex == 0:
                    xShift = -1
                    yShift = -1
                else:
                    xShift = 1
                    yShift = 1
            elif (firstIndex == 1 and secondIndex == 3) or (firstIndex == 3 and secondIndex == 1):
                if firstIndex == 1:
                    xShift = 1
                    yShift = -1
                else:
                    xShift = -1
                    yShift = 1
            
            # === 調整臉框 ===
            if yShift < 0:
                currentMaskFaceBool[-yShift:] = currentMaskFaceBool[:yShift]
                currentMaskFaceBool[:-yShift] = 0
                subFaceMask = currentMaskFaceBool
            elif yShift == 0:
                subFaceMask = currentMaskFaceBool[:]
            elif yShift > 0:
                subFaceMask = currentMaskFaceBool[yShift:]
            
            if xShift < 0:
                subFaceMask[:,-xShift:] = subFaceMask[:,:xShift]
                subFaceMask[:-xShift] = 0
                subFaceMask = subFaceMask[:,:xShift]
            elif xShift == 0:
                subFaceMask = subFaceMask[:,:]
            elif xShift > 0:
                subFaceMask = subFaceMask[:,xShift:]
            
            currentMaskFaceCanvas[:len(subFaceMask),:len(subFaceMask[0])] = subFaceMask
            currentMaskFaceBool = currentMaskFaceCanvas
            prevMatchingRate = matchingRate
            
            loopCount += 1
            
        return xShiftAll, yShiftAll, matchingShow
    
    def findInnerRegion(self,faceRgbImg):
        faceHsvImg = cv2.cvtColor(faceRgbImg, cv2.COLOR_BGR2HSV)
        (channel_H, channel_S, channel_V) = cv2.split(faceHsvImg)
        mask1 = channel_H<30
        mask2 = channel_V>150
        skinMask = mask1 * mask2
        
        firstTruePos1, upDownMask = self.firstTrueMask(skinMask, dirt = "top")
        firstTruePos2, downUpMask = self.firstTrueMask(skinMask, dirt = "bottom")
        firstTruePos3, leftRightMask = self.firstTrueMask(skinMask, dirt = "left")
        firstTruePos4, rightLeftMask = self.firstTrueMask(skinMask, dirt = "right")
        inerMask = upDownMask * downUpMask * leftRightMask * rightLeftMask
        
        return inerMask
    
    def firstTrueMask(self, boolMatrix, dirt):
        
        mask = np.zeros_like(boolMatrix)
        temp = []
        result = []
        truePos = np.where(boolMatrix>0)
        yLIST = list(set(truePos[1]))
        xLIST = list(set(truePos[0]))
        for i, j in zip(*truePos):
            temp.append([i,j])
        
        if dirt == "top":
            temp=sorted(temp, key=lambda x: x[1])
            temp = np.array(temp)
            for item in yLIST:
                firstPoint = list(temp[temp[:,1]==item][0])
                result.append(firstPoint)
                mask[:,firstPoint[1]][firstPoint[0]:] = 1
                
        elif dirt == "left":
            temp = np.array(temp)
            for item in xLIST:
                firstPoint = list(temp[temp[:,0]==item][0])
                result.append(firstPoint)
                mask[firstPoint[0], firstPoint[1]:] = 1
            
        elif dirt == "right":
            temp = np.array(temp)
            for item in xLIST:
                firstPoint = list(temp[temp[:,0]==item][-1])
                result.append(firstPoint)
                mask[firstPoint[0], :firstPoint[1]+1] = 1
            
        elif dirt == "bottom":
            temp=sorted(temp, key=lambda x: x[1])
            temp = np.array(temp)
            for item in yLIST:
                firstPoint = list(temp[temp[:,1]==item][-1])
                result.append(firstPoint)
                mask[:,firstPoint[1]][:firstPoint[0]+1] = 1
                
        else:
            print "Set parameter dirt = 'top', 'bottom, 'left', 'right'"
            
        return result, mask
    
    def csvToList(self,fileName):
        '''
        將 csv 檔轉成 list 格式。
        '''
        csvFILE = open(fileName, "r")
        csvObj = csv.reader(csvFILE)
    
        csvLIST = []
        for item in csvObj:
            csvLIST.append(item)
        csvLIST = np.array(csvLIST)
        
        return csvLIST
    

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

def main(fileName, syllable, geoMatch):
    try:
        csvDICT = CSVtoDICT("./{}".format(fileName))
        for key in csvDICT.keys():
            if os.path.exists("./MP4Files/"+ csvDICT[key]["MP4FILE"]):
                pass
            else:
                print "{} video file load error!".format(csvDICT[key]["MP4FILE"])
                raise SystemExit
    except:
        print "{} file load error!".format(fileName)
        raise SystemExit
    
    # ======== 人工標的眼睛位置 ========
    eyePosLIST = geoMatch.csvToList("./manMadeEyePos.csv")
    eyePosDict = {}
    for item in eyePosLIST:
        eyePosDict['{}'.format(item[0][:4])] = np.array(item[1:],dtype='uint16')
    # ===============================
    
    # ======== 人工標的人臉位置 ========
    facePosLIST = geoMatch.csvToList("./manMade_face_boundingBox.csv")
    facePosDict = {}
    for item in facePosLIST:
        facePosDict['{}'.format(item[0][:4])] = np.array(item[1:],dtype='uint16')
    # ===============================
    
    for frameRate in frameRateLIST:
        print "frame rate =====> ", frameRate
        stepLog = 0
        while stepLog <= (len(csvDICT.keys())-1):
            fileName =csvDICT[stepLog]["MP4FILE"]
            speaker = fileName[:4]
            pureFileName = fileName[:-4]
            cap = cv2.VideoCapture("./MP4Files/"+fileName)
            videoFPS = int(round(cap.get(5)))
            totalFrame = int(cap.get(7))
            step = 120.0/frameRate
            processFrame = np.array(np.arange(0,totalFrame,step),dtype='uint16')
            
            # ===== 偵測第 1 張 frame 中的人臉位置及眼睛位置 =====
            ret, oldRgbFrame = cap.read()
            oldFacePosition = geoMatch.getFacePosition(isFirstFrame=True, videoName=fileName, targetRgbFrame=oldRgbFrame, matchMethod="RMSD")  # , showSiftResult = True
            baseIx, baseIy, baseFx, baseFy = oldFacePosition
            targetFacePosition = oldFacePosition
            
            oldFaceRgbFrame = copy.deepcopy(oldRgbFrame[baseIy:baseFy+1, baseIx:baseFx+1])
            eyePosition = syllable.getEyesPosition(oldFaceRgbFrame, fileName)
            
            if not eyePosition:    # 無法取得當前眼睛位置時，直接以人工標的眼睛位置取代。
                leftEyeX, leftEyeY, rightEyeX, rightEyeY = eyePosDict[speaker]
                ix, iy, fx, fy = facePosDict[speaker]
                eyePosition = leftEyeX-ix, leftEyeY-iy, rightEyeX-ix, rightEyeY-iy
            
            # ===== 準備光流法的參數 winsize 及 灰階影像 =====
            optWinsize = syllable.getOptWinsize(oldFacePosition, eyePosition)
            
            oldGrayFrame = cv2.cvtColor(oldRgbFrame, cv2.COLOR_BGR2GRAY)
            prvsOriginGrayFace = copy.deepcopy(oldGrayFrame[baseIy:baseFy+1,baseIx:baseFx+1])
            prvsSiftGrayFace = copy.deepcopy(oldGrayFrame[baseIy:baseFy+1,baseIx:baseFx+1])
            
            originOptSumLIST = []
            #originOptSquareSumLIST = []
            #siftOptSumLIST = []
            siftOptSquareSumLIST = []
            
            #fourcc = cv2.cv.CV_FOURCC('X','V','I','D')
            #out = cv2.VideoWriter('{}_siftMatch_all_{}fps.avi'.format(pureFileName, frameRate),fourcc, 30.0,((baseFx-baseIx+1)*2,(baseFy-baseIy+1)))
            
            frameID = 1
            while True:
                ret, currentRgbFrame = cap.read()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                if currentRgbFrame == None:
                    break
                if frameID in processFrame:
                    print frameID
                    currentFacePosition = geoMatch.getFacePosition(baseRgbFrame=oldRgbFrame, baseFacePosition=oldFacePosition, targetRgbFrame=currentRgbFrame, matchMethod="RMSD")
                    if not currentFacePosition:
                        print "lack of restrict sift!"     # 無法以 sift 偵測當前人臉位置。人臉位置以前一張 frame 的人臉位置取代。
                    else:
                        targetFacePosition = currentFacePosition
                    ix, iy, fx, fy = targetFacePosition
                    
                    currentSiftRgbFace = copy.deepcopy(currentRgbFrame[iy:fy+1,ix:fx+1])
                    
                    currentGrayFrame = cv2.cvtColor(currentRgbFrame, cv2.COLOR_BGR2GRAY)
                    nextOriginGrayFace = currentGrayFrame[baseIy:baseFy+1,baseIx:baseFx+1]
                    nextSiftGrayFace = currentGrayFrame[iy:fy+1,ix:fx+1]
                    
                    # === 將調整前後的人臉位置以影片方式寫檔 ===
                    #originFaceImg = copy.deepcopy(currentRgbFrame[baseIy:baseFy+1,baseIx:baseFx+1])
                    
                    #cv2.line(originFaceImg, (0,len(originFaceImg)/2), (len(originFaceImg[0])-1,len(originFaceImg)/2), [0,255,0], thickness=1)
                    #cv2.line(originFaceImg, (len(originFaceImg[0])/2,0), (len(originFaceImg[0])/2,len(originFaceImg)-1), [0,255,0], thickness=1)
                    #cv2.line(currentSiftRgbFace, (0,len(currentSiftRgbFace)/2), (len(currentSiftRgbFace[0])-1,len(currentSiftRgbFace)/2), [0,255,0], thickness=1)
                    #cv2.line(currentSiftRgbFace, (len(currentSiftRgbFace[0])/2,0), (len(currentSiftRgbFace[0])/2,len(currentSiftRgbFace)-1), [0,255,0], thickness=1)
                    
                    #stackImg = np.hstack((originFaceImg,currentSiftRgbFace))
                    #cv2.imshow("origin & sift face position", stackImg)
                    #out.write(stackImg)
                    # =============================================
                    
                    # === optical flow ===
                    originFlow = cv2.calcOpticalFlowFarneback(prvsOriginGrayFace,nextOriginGrayFace, 0.5, 3, optWinsize, 3, 5, 1.2, 1)
                    originMag, originAng = cv2.cartToPolar(originFlow[...,0], originFlow[...,1],angleInDegrees=1)
                    originOptSumLIST.append(originMag[len(originMag)/2:].sum())
                    #originOptSquareSumLIST.append(np.square(originMag[len(originMag)/2:]).sum())
                    
                    siftFlow = cv2.calcOpticalFlowFarneback(prvsSiftGrayFace,nextSiftGrayFace, 0.5, 3, optWinsize, 3, 5, 1.2, 1)
                    siftMag, siftAng = cv2.cartToPolar(siftFlow[...,0], siftFlow[...,1],angleInDegrees=1)
                    #siftOptSumLIST.append(siftMag[len(siftMag)/2:].sum())
                    siftOptSquareSumLIST.append(np.square(siftMag[len(siftMag)/2:]).sum())
                    
                    # === flow image show ===
                    #originMag = cv2.cvtColor(np.uint8(originMag*100), cv2.COLOR_GRAY2BGR)
                    #siftMag = cv2.cvtColor(np.uint8(siftMag*100), cv2.COLOR_GRAY2RGB)
                    
                    #stackImg2 = np.hstack((originMag,siftMag))
                    #stackImg3 = np.hstack((stackImg,stackImg2))
                    #cv2.imshow("origin & sift optical flow", stackImg3)
                    
                    # ===== 更新舊矩陣 =====
                    oldRgbFrame = currentRgbFrame
                    oldFacePosition = targetFacePosition
                    
                    prvsOriginGrayFace = nextOriginGrayFace
                    prvsSiftGrayFace = nextSiftGrayFace
                    
                frameID += 1
            #cv2.destroyAllWindows()
            #out.release()
            #os.system("mv {}_siftMatch_all_{}fps.avi {}siftMatchVideo".format(pureFileName, frameRate, outputPath))
            
            maxValue = max(originOptSumLIST)
            originOptSumLIST = [item/float(maxValue) for item in originOptSumLIST]
            maxValue = max(siftOptSquareSumLIST)
            siftOptSquareSumLIST = [item/float(maxValue) for item in siftOptSquareSumLIST]
            
            optPeakPos, optMinPos = syllable.findImageSyllable(fileName, siftOptSquareSumLIST, frameRate)
            audioGroundTruth, audioGroundTruthContent = syllable.getSyllableGroundTruth(fileName)
            syllable.drawOptFigure(fileName, originOptSumLIST, siftOptSquareSumLIST, optPeakPos, optMinPos, audioGroundTruth, frameRate)
            
            # 光流大小寫檔
            outputFILE = open("{}_{}fps_originOpt.csv".format(pureFileName,frameRate), "w")
            w = csv.writer(outputFILE)
            w.writerow(originOptSumLIST)
            outputFILE.close()
            os.system("mv {}_{}fps_originOpt.csv {}optCSV".format(pureFileName, frameRate, outputPath4Opt))
            
            outputFILE = open("{}_{}fps_siftOptSquare_RMSD.csv".format(pureFileName,frameRate), "w")
            w = csv.writer(outputFILE)
            w.writerow(siftOptSquareSumLIST)
            outputFILE.close()
            os.system("mv {}_{}fps_siftOptSquare_RMSD.csv {}optCSV".format(pureFileName, frameRate, outputPath4Opt))
            
            cap.release()
            
            
            
            #========== 影像音節擷取演算法 ==========
            #cap = cv2.VideoCapture("./MP4Files/"+fileName)
            #videoFPS = int(round(cap.get(5)))
            #ret, rgbFrame = cap.read()
    
            #facePosition = syllable.getFacePosition(rgbFrame)
            #if facePosition:
                #faceBbLeftTopX, faceBbLeftTopY, faceBbWidth, faceBbHeight = facePosition
                #faceRgbFrame = copy.deepcopy(rgbFrame[faceBbLeftTopY:faceBbLeftTopY+faceBbHeight,faceBbLeftTopX:faceBbWidth])
                #eyePosition = syllable.getEyesPosition(faceRgbFrame)
                #if eyePosition:
                    #optWinsize = syllable.getOptWinsize()
                    #print "winsize===>", optWinsize
                    
                    #audioSignalPeriod = syllable.getAudioSignalTime(fileName)
                    #imageSyllableLIST = []
                    #for signalPeriod in audioSignalPeriod:
                        #signalBeginFrameID, signalEndFrameID = signalPeriod*videoFPS
                        #frameIdList, optMagList = syllable.calcDenseOpt(fileName,signalBeginFrameID,signalEndFrameID)
                        #syllableLIST = syllable.findImageSyllable(frameIdList, optMagList)
                        #imageSyllableLIST.append(syllableLIST)
                    
            # ===== draw matplotlib figure =====
            stepLog += 1
            print "steplog =====> ", stepLog





if __name__ == "__main__":
    
    testSheetFile = "syllableGroundTruthFile.csv"
    syllable = SpeechSyllable()
    geoMatch = GeometricMatching()
    main(testSheetFile, syllable, geoMatch)


