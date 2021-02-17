import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

############################################################################
#################################Binary#####################################
############################################################################
#Import the images
image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
#create a histogram for the image
plt.hist(image.ravel(),256,[0,256])
#show the histogram
plt.show()
#Import the images
image2 = cv2.imread('image2.jpg', cv2.IMREAD_GRAYSCALE)
#create a histogram for the image
plt.hist(image2.ravel(),256,[0,256])
#show the histogram
plt.show()
#Import the images
image3 = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)
#create a histogram for the image
plt.hist(image3.ravel(),256,[0,256])
#show the histogram
plt.show()

#Blurrying the image using a gaussian filter to get rid of noise
image = cv2.GaussianBlur(image, (5, 5), 0)
image2 = cv2.GaussianBlur(image2, (5, 5), 0)
image3 = cv2.GaussianBlur(image3, (5, 5), 0)

#make a copy of the image so we can reuse it later for Otsu
Im1Bin = image.copy()
#Binary threshold, everything under 100 gets 0, everything over 255
Im1Bin[image > 100] = 255
Im1Bin[image < 100] = 0

#make a copy of the image so we can reuse it later for Otsu
Im2Bin = image2.copy()
#Binary threshold, everything under 60 gets 0, everything over 255
Im2Bin[image2 > 60] = 255
Im2Bin[image2 < 60] = 0

#make a copy of the image so we can reuse it later for Otsu
Im3Bin = image3.copy()
#Binary threshold, everything under 126 gets 0, everything over 255
Im3Bin[image3 > 126] = 255
Im3Bin[image3 < 126] = 0

############################################################################
#################################Otsu#######################################
############################################################################
#methood that return the Otsu threshold value for giver image to avoid copying it 3 times
def getThres(image):
    #initialize variables
    maxVal, thres = 0,0
    height, width = image.shape
    #create a histogram for the image
    his = np.histogram(image, np.array(range(0, 256)))[0]
    #print(his)
    #create an array from 0 to 255 which represents the pixel values
    binVal = np.array(range(0, 256))
    #loop through all the values defined earlier
    for i in binVal[1:-1]:
        #get the value for the first part of the array (0 to i)
        Wb = np.sum(his[:i])/(height*width)
        #get the value for the second part of the array (i to 255)
        Wf = np.sum(his[i:])/(height*width)
        #get the mean for the first part of the array
        meanBack = np.mean(his[:i])
        # get the mean for the second part of the array
        meanFr = np.mean(his[i:])
        #calculate the value to determine the optimal threshold
        newVal = Wb*Wf*(meanBack-meanFr)**2
        #if the optimal value is better, replace it
        if newVal > maxVal:
            maxVal = newVal
            thres = i
    return thres

otsu1 = getThres(image)
print(otsu1)
#Copy the image to apply threshold
Im1Ots = image.copy()
#Apply Otsu threshold on the image
Im1Ots[image > otsu1] = 255
Im1Ots[image < otsu1] = 0

otsu2 = getThres(image2)
print(otsu2)
#Copy the image to apply threshold
Im2Ots = image2.copy()
#Apply Otsu threshold on the image
Im2Ots[image2 > otsu2] = 255
Im2Ots[image2 < otsu2] = 0

otsu3 = getThres(image3)
print(otsu3)
#Copy the image to apply threshold
Im3Ots = image3.copy()
#Apply Otsu threshold on the image
Im3Ots[image3 > otsu3] = 255
Im3Ots[image3 < otsu3] = 0

############################################################################
###########################Display results##################################
############################################################################
titles = ['Original Image 1','BINARY','OTSU','Original Image 2','BINARY','OTSU','Original Image 3','BINARY','OTSU']
images = [image, Im1Bin, Im1Ots, image2, Im2Bin, Im2Ots, image3, Im3Bin, Im3Ots]

#Loop though the previously defined array to display all the images on one plot
for i in range(9):
    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()