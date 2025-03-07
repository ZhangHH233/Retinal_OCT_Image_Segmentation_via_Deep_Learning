import numpy as np
import os
from os import listdir
import sys
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils
import errno
import cv2

import deepModels
############################################################################################################
def readOneImage(imageName, inputTypes, inputPath, outputPath, outputTypes = ''):
    
    img = cv2.imread(inputPath + imageName + inputTypes[0])
    tmp = img[:, :, 0]
    tmp = (tmp - tmp.mean()) / (tmp.std())
    inputLen = len(inputTypes)
    allImg = np.zeros((tmp.shape[0], tmp.shape[1], inputLen))
    allImg[:, :, 0] = tmp    
    
    outputLen = len(outputTypes)
    allOut = np.zeros((tmp.shape[0], tmp.shape[1], outputLen))
    
    if outputPath != '':
        for i in range(1, outputLen):
            img = np.loadtxt(outputPath[1] + imageName + "fdmap" + str(i))
            out = (img - img.mean()) / (img.std())
            allOut[:, :, i-1] = out

        img = cv2.imread(outputPath[0] + imageName + inputTypes[0])
        out = (img[:, :, 0] > 0).astype(np.int_)
        out = np.asarray(out)
        allOut[:, :, outputLen-1] = out
    
    return [allImg, allOut]
############################################################################################################
def readOneDataset(imageNames, inputPath, outputPaths, fd_channel):
    d_names = []
    d_inputs = []
    d_outputs = []
    
    inputTypes = ['.png']

    outputTypes = []
    for i in range (1, fd_channel+1):
        map_name = "fdmap" + str(i)
        outputTypes.append(map_name)
    outputTypes.append('MC')
    
    for fname in imageNames:
        [img, gold] = readOneImage(fname, inputTypes, inputPath, outputPaths, outputTypes)
        d_outputs.append(gold)
        d_inputs.append(img)
        d_names.append(fname)
    
    d_inputs = np.asarray(d_inputs)
    d_outputs = np.asarray(d_outputs)
    return [d_inputs, d_outputs, d_names, outputTypes]
    
############################################################################################################
def listAllImageFiles(imageDirPath, imagePostfix):
    fileList = listdir(imageDirPath)
    postLen = len(imagePostfix)
    imageNames = []
    for i in range(len(fileList)):
        if fileList[i][-postLen::] == imagePostfix:
            imageNames.append(fileList[i][:-postLen])
    return imageNames
############################################################################################################
def createCallbacks(modelFile, earlyStoppingPatience):
    checkpoint = ModelCheckpoint(modelFile, monitor = 'val_loss', verbose = 1, save_best_only = True, 
                                 save_weights_only = True, mode = 'auto', period = 1)
    earlystopping = EarlyStopping(patience = earlyStoppingPatience, monitor = 'val_loss', verbose = 1, 
                                 restore_best_weights = True)
    return [checkpoint, earlystopping]
############################################################################################################
def taskLists(imageNames, inputPath, outputPaths, fd_channel):
    setInputList = []
    setOutputList = []
    [setInputs, setOutputs, setNames, outputTypes] = readOneDataset(imageNames, inputPath, outputPaths, fd_channel)
    
    tSize, im_height, im_width, taskNo = setOutputs.shape
    setInputList.append(setInputs)
    for i in range(taskNo):
        if outputTypes[i] != "MC":
            currOut = setOutputs[:, :, :, i].reshape(tSize, im_height ,im_width, 1)
        else:
            currOut = np_utils.to_categorical(setOutputs[:, :, :, i])
        setOutputList.append(currOut)
        
    return [setInputList, setOutputList]
############################################################################################################
def trainModel(modelFile, trainInputList, trainOutputList, validInputList, validOutputList,
               noOfFeatures, dropoutRate, outputChannelNos = [], 
               lr = 0.001, fd_channel = 1, earlyStoppingPatience = 50, batchSize = 1, maxEpoch = 500):
    
    inputHeight = trainInputList[0].shape[1]
    inputWidth = trainInputList[0].shape[2]
    channelNo = trainInputList[0].shape[3]
    
    model = deepModels.cascaded(inputHeight, inputWidth, channelNo, outputChannelNos[0], outputChannelNos[1],
                                noOfFeatures, dropoutRate, fd_channel, lr)    
    model.summary()
    hist = model.fit(x = trainInputList, y = trainOutputList, validation_data = (validInputList, validOutputList), 
                     validation_split = 0, shuffle = True, batch_size = batchSize, epochs = maxEpoch, verbose = 1, 
                     callbacks = createCallbacks(modelFile, earlyStoppingPatience))
############################################################################################################
def loadModel(modelFile, testInput, noOfFeatures, dropoutRate, fd_channel, outputChannelNos = []):
    inputHeight = testInput.shape[1]
    inputWidth = testInput.shape[2]
    channelNo = testInput.shape[3]

    model = deepModels.cascaded(inputHeight, inputWidth, channelNo, outputChannelNos[0], outputChannelNos[1],
                                noOfFeatures, dropoutRate, fd_channel, lr = 0.1) 
    model.load_weights(modelFile)
    return model
############################################################################################################
def trainUnet(modelFile, trImageNames, trInputPath, trOutputPaths, valImageNames, valInputPath, valOutputPaths,
              outputChannelNos, noOfFeatures, dropoutRate, fd_channel, learningRate):
    
    [trInputList, trOutputList] = taskLists(trImageNames, trInputPath, trOutputPaths, fd_channel)
    [valInputList, valOutputList] = taskLists(valImageNames, valInputPath, valOutputPaths, fd_channel)
    
    trainModel(modelFile, trInputList, trOutputList, valInputList, valOutputList, noOfFeatures, dropoutRate, 
               outputChannelNos, learningRate, fd_channel)
############################################################################################################
def testUnet(modelFile, tsImageNames, tsInputPath, noOfFeatures, dropoutRate, outputChannelNos, fd_channel):
    
    [tsInputs, setOutputs, tsNames, outputTypes] = readOneDataset(tsImageNames, tsInputPath, '', fd_channel)
    model = loadModel(modelFile, tsInputs, noOfFeatures, dropoutRate, fd_channel, outputChannelNos)
    probs = model.predict(tsInputs, batch_size = 4, verbose = 1)
    return [probs, tsNames]
############################################################################################################
def main(trainOrTest, gpuId):
    # Specify the directory paths for the training, validation, and test images as well as
    # for the gold standard images and Fourier descriptor maps
    trInputPath    = '../images/training/'
    # trInputPath    = '../images/training/'
    valInputPath   = '../images/validation/'
    tsInputPath    = '../images/test/'
    outputPath1    = '../golds/'
    outputPath2    = '../fdmaps/'
    imageExtension = '.png'
    
    trImageNames   = listAllImageFiles(trInputPath, imageExtension)
    valImageNames  = listAllImageFiles(valInputPath, imageExtension)
    tsImageNames   = listAllImageFiles(tsInputPath, imageExtension)
    trOutputPath   = [outputPath1, outputPath2]
    valOutputPath  = [outputPath1, outputPath2]
    tsResultPath   = '../test-results/'
    
    # Number of Fourier descriptors
    fd_channel     = 1 
    outChannelNos  = [fd_channel, 2]
    # Specify the model file name
    modelFile      = './models/cascaded_fd' + str(fd_channel) + '.hdf5'
    
    featureNo = [16, 32, 64, 128, 256]
    dropoutR = 0.2
    learningRate = 0.01

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpuId
    
    if trainOrTest == 'tr':
        trainUnet(modelFile, trImageNames, trInputPath, trOutputPath, valImageNames, valInputPath, valOutputPath, 
                  outChannelNos, featureNo, dropoutR, fd_channel, learningRate)
    else:
        [predictions, tsNames] = testUnet(modelFile, tsImageNames, tsInputPath, featureNo, dropoutR, outChannelNos, fd_channel)
        values = predictions[fd_channel][:, :, :, 1]
        for i in range(len(values)):
            fname = tsResultPath + tsNames[i]
            np.savetxt(fname, values[i], fmt = '%1.4f')
###########################################################################################################
if __name__ == "__main__":
    trainOrTest = 'tr'
    gpuId = '2'
    main(trainOrTest, gpuId)

