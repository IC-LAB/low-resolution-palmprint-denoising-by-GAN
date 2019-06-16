# coding:utf-8

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Activation, Dropout
from keras.layers.merge import concatenate
from keras.layers.pooling import AveragePooling2D, GlobalAveragePooling2D
import numpy as np
import cv2
import tensorflow as tf
import math


def Convolution(f, k=3, s=2, border_mode='same', **kwargs):
    """Convenience method for Convolutions."""
    return Conv2D(f, (k, k), padding=border_mode, strides=(s, s))


def Deconvolution(f, output_shape, k=2, s=2, **kwargs):
    """Convenience method for Transposed Convolutions."""
    return Conv2DTranspose(f, (k, k), strides=(s, s), padding='same')


def BatchNorm(mode=2, axis=1, **kwargs):
    """Convenience method for BatchNormalization layers."""
    return BatchNormalization(axis=-1)


def generator(nf, batchSize, name='unet'):
    i = Input(shape=(128, 128, 3))
    # 3 x 128 x 128

    conv1 = Convolution(nf)(i)
    conv1 = BatchNorm(mode=0)(conv1)
    x = LeakyReLU(0.2)(conv1)
    # nf x 64 x 64

    conv2 = Convolution(nf * 2)(x)
    conv2 = BatchNorm(mode=0)(conv2)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 32x 32

    conv3 = Convolution(nf * 4)(x)
    conv3 = BatchNorm(mode=0)(conv3)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 16 x 16

    conv4 = Convolution(nf * 8)(x)
    conv4 = BatchNorm(mode=0)(conv4)
    x = LeakyReLU(0.2)(conv4)
    # nf*8 x 8 x 8

    conv5 = Convolution(nf * 8)(x)
    conv5 = BatchNorm(mode=0)(conv5)
    x = LeakyReLU(0.2)(conv5)
    # nf*8 x 4 x 4

    conv6 = Convolution(nf * 8)(x)
    conv6 = BatchNorm(mode=0)(conv6)
    x = LeakyReLU(0.2)(conv6)
    # nf*8 x 2 x 2

    conv7 = Convolution(nf * 8, k=2, s=1, border_mode='valid')(x)
    conv7 = BatchNorm(mode=0)(conv7)
    x = LeakyReLU(0.2)(conv7)
    # nf*8 x 1 x 1

    deconv1 = Deconvolution(nf * 8, (None, 2, 2, nf * 8), k=2, s=2)(x)
    deconv1 = BatchNorm(mode=0)(deconv1)
    deconv1 = Dropout(0.5)(deconv1)
    x = concatenate([deconv1, conv6], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*8 + nf*8) x 2 x 2

    deconv2 = Deconvolution(nf * 8, (None, 4, 4, nf * 8))(x)
    deconv2 = BatchNorm(mode=0)(deconv2)
    deconv2 = Dropout(0.5)(deconv2)
    x = concatenate([deconv2, conv5], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*8 + nf*8) x 4 x 4

    deconv3 = Deconvolution(nf * 8, (None, 8, 8, nf * 8))(x)
    deconv3 = BatchNorm(mode=0)(deconv3)
    deconv3 = Dropout(0.5)(deconv3)
    x = concatenate([deconv3, conv4], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*8 + nf*8) x 8 x 8

    deconv4 = Deconvolution(nf * 8, (None, 16, 16, nf * 8))(x)
    deconv4 = BatchNorm(mode=0)(deconv4)
    x = concatenate([deconv4, conv3], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*8 + nf*8) x 16 x 16

    deconv5 = Deconvolution(nf * 4, (None, 32, 32, nf * 4))(x)
    deconv5 = BatchNorm(mode=0)(deconv5)
    x = concatenate([deconv5, conv2], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*4 + nf*4) x 32 x 32

    deconv6 = Deconvolution(nf * 2, (None, 64, 64, nf * 2))(x)
    deconv6 = BatchNorm(mode=0)(deconv6)
    x = concatenate([deconv6, conv1], axis=-1)
    x = LeakyReLU(0.2)(x)
    # (nf*2 + nf*2) x 64 x 64

    deconv7 = Deconvolution(3, (None, 128, 128, 3))(x)
    # 3 x 128 x 128

    out = Activation('tanh')(deconv7)

    unet = Model(i, out, name=name)

    return unet


def discriminator(nf, opt=RMSprop(lr=0.00005), name='d'):
    i = Input(shape=(128, 128, 3 + 3))
    # (3 + 3) x 128 x 128

    conv1 = Convolution(nf * 1)(i)
    x = LeakyReLU(0.2)(conv1)
    # nf*1 x 64x 64

    conv2 = Convolution(nf * 2)(x)
    x = LeakyReLU(0.2)(conv2)
    # nf*2 x 32 x 32

    conv3 = Convolution(nf * 4)(x)
    x = LeakyReLU(0.2)(conv3)
    # nf*4 x 16 x 16

    conv4 = Convolution(1)(x)
    # 1 x 8 x 8

    out = GlobalAveragePooling2D()(conv4)

    d = Model(i, out, name=name)

    def d_loss(y_true, y_pred):
        return K.mean(y_true * y_pred)

    d.compile(optimizer=opt, loss=d_loss)
    return d


# build Gabor filters
def buildGaborFilters(kSize):
    filters = np.zeros(shape=(kSize, kSize, 6))
    cha = 0
    for theta in np.arange(0, np.pi, np.pi / 6):
        kern = cv2.getGaborKernel(ksize=(kSize, kSize), sigma=4.6, theta=theta, lambd=(np.pi / 2.0), gamma=0.5, psi=0,
                                  ktype=cv2.CV_32F)
        kern /= 1.5 * kern.sum()
        # filters.append(kern)
        filters[:, :, cha] = kern
        cha = cha + 1
    filters = np.asarray(filters, np.float32)
    filters = np.expand_dims(filters, axis=2)
    # print(filters.shape)
    filters = np.concatenate((np.concatenate((filters, filters), axis=2), filters), axis=2)
    # print(filters.shape)
    return filters


# generate Gabor filters
def genGaborFilter():
    filter = np.zeros(shape=(35, 35, 1, 6), dtype=np.float32)

    # parameters
    fSigma = 4.6
    f2Delta = 2.6
    g_iFilterSize = 35

    g_iHalfFilterSize = math.floor(g_iFilterSize / 2)
    PI = 3.14159265
    ln2_2 = np.sqrt(2 * np.log(2))
    fK = ln2_2 * (f2Delta + 1) / (f2Delta - 1)
    fW0 = fK / fSigma
    fFactor1 = fW0 / (np.sqrt(2 * PI) * fK)
    fFactor2 = -(fW0 * fW0) / (8 * fK * fK)

    for k in np.arange(0, 6):
        fAngle = k * PI / 6
        fSin = np.sin(fAngle)
        fCos = np.cos(fAngle)

        pFilter = np.zeros(shape=(1, 35 * 35), dtype=np.float32)
        for i in np.arange(0, g_iFilterSize):
            x = i - g_iHalfFilterSize
            for j in np.arange(0, g_iFilterSize):
                y = j - g_iHalfFilterSize
                x1 = x * fCos + y * fSin
                y1 = y * fCos - x * fSin
                fTemp = fFactor1 * np.exp(fFactor2 * (4 * x1 * x1 + y1 * y1))
                pFilter[0, j * g_iFilterSize + i] = fTemp * np.cos(fW0 * x * fCos + fW0 * y * fSin)
        pFilter = pFilter - np.mean(pFilter)

        filter[:, :, 0, k] = np.reshape(pFilter, newshape=(35, 35))

    return filter


# get filtered img
def gaborFilter(batchImg, filters):
    filters_tensor = tf.convert_to_tensor(filters)
    filters_max = K.max(filters_tensor)
    accum = K.conv2d(batchImg[:, :, :, 0:1], kernel=filters_tensor, padding='same', data_format='channels_last')
    accum = accum/filters_max

    return accum


def pix2pix(atob, d, alpha=0.1, belta=0.01, opt=RMSprop(lr=0.00005), name='pix2pix'):
    a = Input(shape=(128, 128, 3))
    b = Input(shape=(128, 128, 3))

    # A -> B'
    bp = atob(a)

    # Discriminator receives the pair of images
    d_in = concatenate([a, bp], axis=-1)

    pix2pix = Model([a, b], d(d_in), name=name)

    def pix2pix_loss(y_true, y_pred):

        # Adversarial Loss
        L_adv = K.mean(y_true * y_pred)

        # A to B loss
        b_flat = K.batch_flatten(b)
        bp_flat = K.batch_flatten(bp)
        L_atob = K.mean(K.abs(b_flat - bp_flat))

        # Gabor loss
        filters = genGaborFilter()
        filteredB = gaborFilter(batchImg=b, filters=filters)
        filteredB_flat = K.batch_flatten(filteredB)
        filteredBp = gaborFilter(batchImg=bp, filters=filters)
        filteredBp_flat = K.batch_flatten(filteredBp)
        L_gabor = K.mean(K.abs(filteredB_flat - filteredBp_flat))

        return L_adv + alpha * L_atob + belta * L_gabor

    # This network is used to train the generator. Freeze the discriminator part.
    pix2pix.get_layer('d').trainable = False

    pix2pix.compile(optimizer=opt, loss=pix2pix_loss)
    return pix2pix
