import numpy as np
from matplotlib import pyplot as plt
import cv2
import pandas as pd
import time
import seaborn as sns
import os

def get_image_list(imageDirectory):
    """
    Don't be a jerk, add docs
    """
    print("Searching in :",imageDirectory)
    imageList = []
    for root, dirs, files in os.walk(imageDirectory, topdown=False):
        for name in files:
            if '.png' in name:
                imageList.append(os.path.join(root, name))
    print('Found {} .png images.'.format(len(imageList)))
    imageList.sort()
    return imageList

def show_image_file(file):
    img = cv2.imread(file)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15,8))
    plt.imshow(imgRGB)
    plt.show()
    
def show_image(image, cmap = 'default'):
    plt.figure(figsize=(15,8))
    if cmap != 'default':
        plt.imshow(image, cmap = cmap)
    else:
        plt.imshow(image)
    plt.show()

def plot_image_file_histogram(file):
    img = cv2.imread(file)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        ax[1].plot(histr,color = col)
        ax[1].set_xlim([0,256])
    ax[1].grid()
    ax[1].set_xlabel("Color Value")
    ax[1].set_ylabel("Pixel Count")
    ax[0].imshow(imgRGB)
    plt.show()
    
def get_color_averages(imageList):
    imageDict = {'file':[], 'saturationMean':[], 'luminanceMean':[], 'hueMean':[],
                   'blueMean':[], 'greenMean':[], 'redMean':[]}
    start = time.time()
    for i, file in enumerate(imageList):
        elapsed = time.time() - start
        print("Processing Image {}/{} - Time Elapsed {} seconds".format(i, len(imageList), round(elapsed,2)), end='\r')
        imageDict['file'].append(file)
        img = cv2.imread(file)
        imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
        imageDict['hueMean'].append(np.mean(imgHLS[:,:,0]))
        imageDict['luminanceMean'].append(np.mean(imgHLS[:,:,1]))
        imageDict['saturationMean'].append(np.mean(imgHLS[:,:,2]))
        imageDict['blueMean'].append(np.mean(img[:,:,0]))
        imageDict['greenMean'].append(np.mean(img[:,:,1]))
        imageDict['redMean'].append(np.mean(img[:,:,2]))
    return pd.DataFrame(imageDict)

def plot_image_file_histogram_bgr(file):
    img = cv2.imread(file)
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(2, 2, figsize=(15,10))
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        ax[1][1].plot(histr,color = col)
        ax[1][1].set_xlim([0,256])
    ax[1][1].grid()
    ax[1][1].set_xlabel("Color Value")
    ax[1][1].set_ylabel("Pixel Count")
    ax[1][1].legend(['Blue', 'Green', 'Red'])
    imgRed = imgRGB.copy()
    imgRed[:,:,1] = 0
    imgRed[:,:,2] = 0
    imgGreen = imgRGB.copy()
    imgGreen[:,:,0] = 0
    imgGreen[:,:,2] = 0
    imgBlue = imgRGB.copy()
    imgBlue[:,:,0] = 0
    imgBlue[:,:,1] = 0
    ax[0][0].imshow(imgBlue)
    ax[0][1].imshow(imgGreen)
    ax[1][0].imshow(imgRed)
    ax[0][0].set_title('Blue')
    ax[0][1].set_title('Green')
    ax[1][0].set_title('Red')
    plt.show()

def plot_image_file_histogram_hls(file):
    img = cv2.imread(file)
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
    fig, ax = plt.subplots(2, 2, figsize=(15,10))
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([imgHLS],[i],None,[256],[0,256])
        ax[1][1].plot(histr,color = col)
        ax[1][1].set_xlim([0,256])
    ax[1][1].grid()
    ax[1][1].set_xlabel("Color Value")
    ax[1][1].set_ylabel("Pixel Count")
    ax[1][1].legend(['Hue', 'Luminance', 'Saturation'])
    ax[0][0].imshow(imgHLS[:,:,0], cmap='gray')
    ax[0][1].imshow(imgHLS[:,:,1], cmap='gray')
    ax[1][0].imshow(imgHLS[:,:,2], cmap='gray')
    ax[0][0].set_title('Hue')
    ax[0][1].set_title('Luminance')
    ax[1][0].set_title('Saturation')
    plt.show()

def seperate_night_day(imageDirectory, threshold=70):
    imageList = []
    for root, dirs, files in os.walk(imageDirectory, topdown=False):
        for name in files:
            if '.jpg' in name:
                imageList.append(os.path.join(root, name))
    print('Found {} .jpg images.'.format(len(imageList)))
    imageList.sort()
    preMoveDict = {'file':[], 'saturationMean':[], 'luminanceMean':[], 'hueMean':[],
                   'blueMean':[], 'greenMean':[], 'redMean':[]}
    start = time.time()
    for i, file in enumerate(imageList):
        elapsed = time.time() - start
        print("Processing Image {}/{} - Time Elapsed {} seconds".format(i, len(imageList), round(elapsed,2)), end='\r')
        preMoveDict['file'].append(file)
        img = cv2.imread(file)
        imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
        preMoveDict['hueMean'].append(np.mean(imgHLS[:,:,0]))
        preMoveDict['luminanceMean'].append(np.mean(imgHLS[:,:,1]))
        preMoveDict['saturationMean'].append(np.mean(imgHLS[:,:,2]))
        preMoveDict['blueMean'].append(np.mean(img[:,:,0]))
        preMoveDict['greenMean'].append(np.mean(img[:,:,1]))
        preMoveDict['redMean'].append(np.mean(img[:,:,2]))
    preMoveDf = pd.DataFrame(preMoveDict)
    preMoveDf['goodImage'] = preMoveDf['luminanceMean']>threshold
    try:
        os.makedirs(imageDirectory+'/night')
        print("Created Directory: " + imageDirectory+'/night')
    except FileExistsError:
        print("Path already exsists: " + imageDirectory+'/night')
    try:
        os.makedirs(imageDirectory+'/day')
        print("Created Directory: " + imageDirectory+'/day')
    except FileExistsError:
        print("Path already exsists: " + imageDirectory+'/day')

    for i in preMoveDf.index:
        file = preMoveDf['file'][i]
        splitPathList = file.split('/')
        if preMoveDf['goodImage'][i]:
            try:
                splitPathList[-2] = splitPathList[-2]+'/day'
                newPath = '/'.join(splitPathList)
                os.rename(file, newPath)
            except FileNotFoundError:
                print("File Not Found: " + file)

        else:
            try:
                splitPathList[-2] = splitPathList[-2]+'/night'
                newPath = '/'.join(splitPathList)
                os.rename(file, newPath)  
            except FileNotFoundError:
                print("File Not Found: " + file)

def create_timelapse(folder, name='timelapse', frameRate=20):
    image_folder = folder
    video_name = folder+'/'+name+'.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    images.sort()
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, frameRate, (width,height))

    for i, image in enumerate(images):
        print("Adding image to video: {}/{}".format(i+1, len(images)), end='\r')
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

def plot_gray_histogram(img):
    plt.figure(figsize=(15,5))
    plt.hist(img.ravel(),256,[0,256])
    plt.grid()
    plt.show()

def timing_printer(text, start, timingList):
    elapsed = time.time()-start
    if len(timingList)!=0:
        elapsed = elapsed - timingList[-1][1]
    print(text + ": {} seconds".format(round(elapsed, 3)))
    return (text, elapsed)

def separate_seed_trays_step1(file, timing = False):
    timingList = []
    start = time.time()
    img = cv2.imread(file)
    if timing: timingList.append(timing_printer("Read Image", start, timingList))
    imgHLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS_FULL)
    if timing: timingList.append(timing_printer("Convert to HLS", start, timingList))
    imgHue = imgHLS[:,:,0]
    imgHueBlurred = cv2.blur(imgHue, (50,50))
    if timing: timingList.append(timing_printer("Blur Image", start, timingList))
    equ = cv2.equalizeHist(imgHueBlurred)
    if timing: timingList.append(timing_printer("Equalize Hue Histogram", start, timingList))
    mean = np.mean(equ)
    ret, thresh1 = cv2.threshold(equ, 0.75*mean, 255, cv2.THRESH_BINARY_INV)
    if timing: timingList.append(timing_printer("Create Mask for Saturation", start, timingList))
    maskedSaturation = np.bitwise_and(imgHLS[:,:,2], thresh1)
    maskedSaturationBlurred = cv2.blur(maskedSaturation, (5,5))
    equSaturation = cv2.equalizeHist(maskedSaturationBlurred)
    meanSaturation = np.mean(equSaturation)
    ret, threshSaturation = cv2.threshold(equSaturation, meanSaturation*2, 255, cv2.THRESH_TOZERO)
    ret, threshSaturation = cv2.threshold(threshSaturation, 200, 255, cv2.THRESH_TOZERO_INV)
    ret, threshSaturation = cv2.threshold(threshSaturation, 1, 255, cv2.THRESH_BINARY)
    if timing: timingList.append(timing_printer("Threshold Saturation", start, timingList))
    kernel = np.ones((3,3), dtype=np.uint8)
    dilated = cv2.dilate(threshSaturation,kernel,iterations = 10)
    if timing: timingList.append(timing_printer("Dilate Saturation", start, timingList))
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    externalContours = np.zeros(dilated.shape)
    for i in range(len(contours)):
        # external contours
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(externalContours, contours, i, 255, -1)
    if timing: timingList.append(timing_printer("Analyze Contours", start, timingList))
    kernel = np.ones((3,3), dtype=np.uint8)
    opening = cv2.morphologyEx(externalContours, cv2.MORPH_OPEN, kernel, iterations=20)
    if timing: timingList.append(timing_printer("Morphological Opening", start, timingList))
    dilation = cv2.dilate(opening,kernel,iterations = 5)
    if timing: timingList.append(timing_printer("Morphological Dilation", start, timingList))
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=30)
    if timing: timingList.append(timing_printer("Morphological Closing", start, timingList))
    hueDistanceTransform = cv2.distanceTransform(closing.astype(np.uint8), cv2.DIST_L2, 5)
    ret, distanceThresh = cv2.threshold(hueDistanceTransform, 0.75*np.max(hueDistanceTransform), 255, cv2.THRESH_BINARY)
    if timing: timingList.append(timing_printer("Distance Transform Analysis", start, timingList))
    if timing:
        return distanceThresh, timingList
    else:
        return distanceThresh

if __name__ == '__main__':
    print("OK")
