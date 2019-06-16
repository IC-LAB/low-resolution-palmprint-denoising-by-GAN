# coding:utf-8
# generate data use trained GAN

import os
import numpy as np
import model as m
from tqdm import tqdm
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import copy
from PIL import Image
import pickle
import shutil

iDir = '../../result/gaborLossNorm'
lossDir = iDir + '/loss'
weightDir = iDir + '/weights'
generateDir = '../../data/5Noise/test'

if (os.path.exists(generateDir+'/gen')):
    shutil.rmtree(generateDir+'/gen')
os.makedirs(generateDir+'/gen')

# lr
lr = 0.00005
natob = 32
nfd = 16

# Generator
generator = m.generator(natob, batchSize=64)
generator.compile(loss='binary_crossentropy', optimizer="SGD")
generator.load_weights(weightDir + '/generator_10000')

fileList = os.listdir(generateDir + '/x5')
len = len(fileList)

for idx in tqdm(range(1,len+1)):
    # load the input data
    imgX = Image.open(generateDir+'/x5/'+str(idx)+'.bmp').convert('RGB')
    testX = (np.expand_dims(np.asarray(imgX, np.float32), axis=0) - 127.5) / 127.5
    # generate the output image
    generateY = generator.predict(testX)
    # save the output image
    imgY = generateY[0,:,:,0]
    imgY = (imgY* 127.5) + 127.5
    imgY = Image.fromarray(np.uint8(imgY))
    imgY.save(generateDir+'/gen/'+str(idx)+'.bmp')
