import numpy as np
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from tensorflow.keras import optimizers

############################################################################################################
def unetOneBlock(blockInput, noOfFeatures, filterSize, dropoutRate):
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same',  
                                kernel_initializer = 'he_uniform')(blockInput)
    blockOutput = Dropout(dropoutRate)(blockOutput)
    blockOutput = Convolution2D(noOfFeatures, filterSize, activation = 'relu', padding = 'same', 
                                kernel_initializer = 'he_uniform')(blockOutput)
    return blockOutput
############################################################################################################
def unetOneEncoderBlock(blockInput, noOfFeatures, filterSize, dropoutRate):
    conv = unetOneBlock(blockInput, noOfFeatures, filterSize, dropoutRate)
    pool = MaxPooling2D(pool_size = (2, 2))(conv)
    return [conv, pool]
############################################################################################################
def unetOneDecoderBlock(blockInput, longSkipInput, noOfFeatures, filterSize, dropoutRate):
    upR = concatenate([UpSampling2D(size = (2, 2))(blockInput), longSkipInput], axis = 3)
    conv = unetOneBlock(upR, noOfFeatures, filterSize, dropoutRate)
    return conv
############################################################################################################
def oneEncoderPath(inputs, noOfFeatures, filterSize, dropoutRate):
    
    [conv1, pool1] = unetOneEncoderBlock(inputs, noOfFeatures[0], filterSize, dropoutRate)
    [conv2, pool2] = unetOneEncoderBlock(pool1,  noOfFeatures[1], filterSize, dropoutRate)
    [conv3, pool3] = unetOneEncoderBlock(pool2,  noOfFeatures[2], filterSize, dropoutRate)
    [conv4, pool4] = unetOneEncoderBlock(pool3,  noOfFeatures[3], filterSize, dropoutRate)

    return [conv1, conv2, conv3, conv4, pool4]
############################################################################################################
def oneDecoderPath(lastConv, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4):

    deconv4 = unetOneDecoderBlock(lastConv, conv4, noOfFeatures[3], filterSize, dropoutRate)
    deconv3 = unetOneDecoderBlock(deconv4, conv3, noOfFeatures[2], filterSize, dropoutRate)
    deconv2 = unetOneDecoderBlock(deconv3, conv2, noOfFeatures[1], filterSize, dropoutRate)
    deconv1 = unetOneDecoderBlock(deconv2, conv1, noOfFeatures[0], filterSize, dropoutRate)
    
    return deconv1
############################################################################################################
def CasUNet(inputs, outputNo, noOfFeat, fsize, drate, outName):

    [conv1, conv2, conv3, conv4, poolR] = oneEncoderPath(inputs, noOfFeat, fsize, drate)
    lastConv = unetOneBlock(poolR, noOfFeat[4], fsize, drate)
    lastDeconv = oneDecoderPath(lastConv, noOfFeat, fsize, drate, conv1, conv2, conv3, conv4)

    if outputNo == 2:
        netOutput = Convolution2D(outputNo, (1, 1), activation = 'softmax', name = outName)(lastDeconv)
        outputLoss = "categorical_crossentropy"

    elif outputNo == 1:
        netOutput = Convolution2D(outputNo, (1, 1), activation = 'linear', name = outName)(lastDeconv)
        outputLoss = "mean_squared_error"

    return netOutput, outputLoss
############################################################################################################
def cascaded(inputHeight, inputWidth, channelNo, interOutputNo, finalOutputNo, noOfFeatures, 
             dropoutRate, fd_channel, lr = 0.01):
    filterSize = (3, 3)
    optimizer = optimizers.Adadelta(learning_rate = lr)
    
    inputs = Input(shape = (inputHeight, inputWidth, channelNo), name = 'input')
    [conv1, conv2, conv3, conv4, poolR] = oneEncoderPath(inputs, noOfFeatures, filterSize, dropoutRate)
    lastConv = unetOneBlock(poolR, noOfFeatures[4], filterSize, dropoutRate)
    
    lastDeconv = []
    for i in range(fd_channel):
        lastDeconv.append(oneDecoderPath(lastConv, noOfFeatures, filterSize, dropoutRate, conv1, conv2, conv3, conv4))
        
    interOutput = []
    for i in range(fd_channel):
        interOutput.append(Convolution2D(interOutputNo, (1, 1), activation = 'linear', name = "interO" + str(i+1))(lastDeconv[i]))
    
    losses = []
    for i in range(fd_channel):
        losses.append("mean_squared_error")

    interOutput.insert(0,inputs)
    inputs2 = concatenate(interOutput, axis = 3)
    finalOutput, finalLoss = CasUNet(inputs2, finalOutputNo, noOfFeatures, filterSize, dropoutRate, 'finalO')  

    interOutput.append(finalOutput)
    model = Model(inputs = inputs, outputs = interOutput[1:])
     
    losses.append(finalLoss)
    model.compile(loss = losses, optimizer = optimizer, metrics = ['mean_squared_error'])
    return model
