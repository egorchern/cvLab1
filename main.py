import cv2
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt
def convolve(img, kernel):
    (rows, columns) = img.shape
    kernelSize = kernel.shape[0]
    # pad array
    resultImg = np.empty(shape=(rows, columns), dtype=np.uint8)
    padSize = int((kernelSize - 1) / 2)
    intermediate = np.pad(img, pad_width=((padSize, padSize), (padSize, padSize)), constant_values=0)
    # Iterate through pixels
    # Modify indexes to go through non pad values
    for rowI in range(padSize, rows + padSize):
        for columnI in range(padSize, columns + padSize):
            # Get kernel sized slice
            imgSlice = intermediate[rowI - padSize:rowI + padSize + 1, columnI - padSize:columnI + padSize + 1]
            mulRes = np.multiply(imgSlice, kernel)
            resultImg[rowI - padSize, columnI - padSize] = mulRes
    return resultImg

def threshold(img, threshold):
    MAX_VAL = 255
    MIN_VAL = 0
    (rows, columns) = img.shape
    resultImg = np.empty(img.shape, dtype=np.uint8)
    for rowI in range(rows):
        for columnI in range(columns):
            val = img[rowI][columnI]
            resultImg[rowI][columnI] = MAX_VAL if val >= threshold else MIN_VAL
            
    

def getGradientMagnitudes(horizontalEdges, verticalEdges):
    resultImg = np.empty(shape=horizontalEdges.shape, dtype=np.uint8)
    (rows, columns) = horizontalEdges.shape
    for rowI in range(rows):
        for columnI in range(columns):
            horizontalMagnitude = horizontalEdges[rowI][columnI]
            verticalMagnitude = verticalEdges[rowI][columnI]
            resultImg[rowI][columnI] = min(255, sqrt(horizontalMagnitude ** 2 + verticalMagnitude ** 2))
    return resultImg

def showHistogram(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.reshape(256)
    plt.bar(np.linspace(0,255,256), hist)
    plt.title('Histogram')
    plt.ylabel('Frequency')
    plt.xlabel('Grey Level')
    plt.show()

def getEdges(useWeighted = False):
    prewitHorizontalKernel = np.asanyarray([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1 ]
    ], dtype=np.float32)
    prewitVerticalKernel = np.asanyarray([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1 ]
    ], dtype=np.float32)
    sobelHorizontalKernel = np.asanyarray([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1 ]
    ], dtype=np.float32)
    sobelVerticalKernel = np.asanyarray([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1 ]
    ], dtype=np.float32)
    horizontalEdges = convolve(img, prewitHorizontalKernel)
    horizontalEdges = np.maximum(np.abs(horizontalEdges, dtype=np.uint8), np.full(horizontalEdges.shape, 255, dtype=np.uint8))
    verticalEdges = convolve(img, prewitVerticalKernel)

def main ():
    input_file = 'kitty.bmp'
    img = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    # Check for success
    if img is None:
        print('Failed to open', input_file)
        return
    
    
    edgeMagnitudes = getGradientMagnitude(horizontalEdges, verticalEdges)
    windowName1 = "Display"
    windowName2 = "Display2"
    cv2.namedWindow(windowName1, cv2.WINDOW_NORMAL)
    cv2.namedWindow(windowName2, cv2.WINDOW_NORMAL)
    thresholded = threshold(edgeMagnitudes, 140)
    thresholded.dtype = np.uint8
    while True:
        if cv2.waitKey(1) == ord(' '):
            break

    cv2.imshow(windowName1, horizontalEdges)
    cv2.imshow(windowName2, verticalEdges)
    showHistogram(edgeMagnitudes)
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()