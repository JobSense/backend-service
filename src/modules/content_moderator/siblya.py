# from __future__ import division, print_function
import time
import numpy as np
import math
import os
import json
import sys
import re
import pickle
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict

import pandas as pd
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import h5py

import keras
from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Merge
from keras.models import load_model

from flask import current_app


BATCH_SIZE = 64
VOCAB_SIZE = 79000
SEQ_LEN_TITLE = 40
SEQ_LEN_PROS = 60
SEQ_LEN_CONS = 120
SEQ_LEN_LOC = 20
WEIGHT = [1, 1]


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        # todo: check that this is correct
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class Attention(Layer):
    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True,
                 return_attention=False,
                 **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Note: The layer has been tested with Keras 1.x
        Example:

            # 1
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
            # next add a Dense layer (for classification/regression) or whatever...
            # 2 - Get the attention scores
            hidden = LSTM(64, return_sequences=True)(words)
            sentence, word_scores = Attention(return_attention=True)(hidden)
        """
        self.supports_masking = True
        self.return_attention = return_attention
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((1,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        eij = dot_product(x, self.W)

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        weighted_input = x * K.expand_dims(a)

        result = K.sum(weighted_input, axis=1)

        if self.return_attention:
            return [result, a]
        return result

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [(input_shape[0], input_shape[-1]),
                    (input_shape[0], input_shape[1])]
        else:
            return input_shape[0], input_shape[-1]


# the input review should be a dictionary contain title, pros, cons and workLocation . e.g. {"id": "1234567", "title" : "nice company", "pros" : "a lot of benifits, people are good", "cons" : "too far away from city", "workLocation" : "Melbourne, Australia", "kfcPred" : 0 }, premodel_file is the dir path for the premodel


def spremodel_load(premodel_file):
    with open(premodel_file, 'rb') as f:
        premodel = pickle.load(f)
    return premodel


def review2inx(review, premodel):
    # load premodel

    word2idx = premodel['word2idx']
    idx2word = premodel['idx2word']
    idx2label = premodel['idx2label']

    p_mark = re.compile('([\$\@\%]+)')
    p_rm = re.compile('([^a-zA-Z0-9\$\@\%]+)')

    # remember to modify/remove the encoding
    review_inx = {}
    review_inx['title_apply_x'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(
        w) is not None else VOCAB_SIZE - 1 for w in p_rm.sub(' ', p_mark.sub(r' \1 ', review.get('title').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=SEQ_LEN_TITLE, value=0)
    review_inx['pros_apply_x'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(
        w) is not None else VOCAB_SIZE - 1 for w in p_rm.sub(' ', p_mark.sub(r' \1 ', review.get('pros').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=SEQ_LEN_PROS, value=0)
    review_inx['cons_apply_x'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(
        w) is not None else VOCAB_SIZE - 1 for w in p_rm.sub(' ', p_mark.sub(r' \1 ', review.get('cons').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=SEQ_LEN_CONS, value=0)
    review_inx['location_apply_x'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(
        w) is not None else VOCAB_SIZE - 1 for w in p_rm.sub(' ', p_mark.sub(r' \1 ', review.get('workLocation').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=SEQ_LEN_LOC, value=0)
    review_inx['kfc_apply_x'] = sequence.pad_sequences(
        [np.array(review.get('kfcPred')).reshape(1)], maxlen=1, value=0)
    review_inx['id_apply'] = [review.get('id')]

    return review_inx


def smodel_load(model_file):
    model = load_model(model_file, custom_objects={'Attention': Attention})
    return model


def scoring(review_inx, model):

    preds = model.predict([review_inx['title_apply_x'], review_inx['pros_apply_x'],
                           review_inx['cons_apply_x'], review_inx['location_apply_x'], review_inx['kfc_apply_x']])

    preds_normalzied = (preds * WEIGHT) / np.sum(preds *
                                                 WEIGHT, axis=1).reshape(preds.shape[0], 1)
    class_idxs = np.argmax(preds_normalzied, axis=1)

    pred_output = {}
    pred_output['id'] = review_inx['id_apply'][0]
    pred_output['isBlocked'] = 'yes' if class_idxs[0] == 1 else 'no'
    pred_output['score'] = preds_normalzied[0, 1]

    return pred_output


# fill in the model file name and path here
# the model file is in s3a://incubator-data-science-773480812817-ap-southeast-1/models/comment_review/comment_review_au/comment_main_dnn_dev_kfc_location_model_73.h5
# the pre- model file is in s3a://incubator-data-science-773480812817-ap-southeast-1/models/comment_review/comment_review_au/comment_au_main_dev_kfc_location_pre_model_73.pkl

def handler(review):
    """
    Review payload schema:
        {
            "id": "1234567",
            "title": "good company",
            "pros": "the people are friendly, a lot of benefits",
            "cons": "a lot of bullying managers",
            "workLocation": "Australia",
            "kfcPred": 0
        }
    """
    review_input = review2inx(review, current_app.pre_model)
    score = scoring(review_input, current_app.model)

    return score
