from __future__ import division,print_function
import numpy as np
import math, os, json, sys, re
import pickle
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict

import pandas as pd
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis


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

VOCAB_SIZE = 90000
job_title_len = 11
job_seniority_level_len = 2
job_industry_len = 6
job_description_len = 800
job_requirement_len = 300
job_employment_type_len = 2
company_name_len = 8
company_size_len = 4
job_specializations_len = 7
job_roles_len = 8
job_work_locations_len = 13
company_location_len = 5
qualification_code_len = 20
field_of_study_len = 14
mandatory_skill_keyword_len = 26


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
        # and this results in NaN's. A workaround is to add a very small positive number epsilon to the sum.
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


# the input review should be a dictionary contain title, pros, cons and work_location . e.g. {"id": "1234567", "title" : "nice company", "pros" : "a lot of benifits, people are good", "cons" : "too far away from city", "work_location" : "Melbourne, Australia", "kfc_pred" : 0 }, premodel_file is the dir path for the premodel

def jobads2inx(jobads, premodel):
  #load premodel

  word2idx = premodel['word2idx']
  idx2word = premodel['idx2word']
  
  p_mark = re.compile('([\$\@]+)')
  p_rm = re.compile('([^a-zA-Z0-9\$\@]+)')
  
  #remember to modify/remove the encoding
  jobads_inx = {}
  
  jobads_inx['job_title'] =  sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('job_title').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=job_title_len, value=0)
  
  jobads_inx['job_seniority_level'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('job_seniority_level').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=job_seniority_level_len, value=0)
  
  jobads_inx['job_industry'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('job_industry').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=job_industry_len, value=0)
  
  jobads_inx['job_description'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('job_description').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=job_description_len, value=0)
      
  jobads_inx['job_requirement'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('job_requirement').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=job_requirement_len, value=0)
  
  jobads_inx['job_employment_type'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('job_employment_type').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=job_employment_type_len, value=0)
  
  jobads_inx['company_name'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('company_name').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=company_name_len, value=0)
            
  jobads_inx['company_size'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('company_size').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=company_size_len, value=0)
  
  jobads_inx['job_specializations'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('job_specializations').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=job_specializations_len, value=0)
  
  jobads_inx['job_roles'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('job_roles').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=job_roles_len, value=0)
  
  jobads_inx['job_work_locations'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('job_work_locations').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=job_work_locations_len, value=0)
  
  jobads_inx['company_location'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('company_location').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=company_location_len, value=0)
    
  jobads_inx['qualification_code'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('qualification_code').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=qualification_code_len, value=0)
      
  jobads_inx['field_of_study'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('field_of_study').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=field_of_study_len, value=0)
        
  jobads_inx['mandatory_skill_keyword'] = sequence.pad_sequences([np.array([word2idx.get(w) if word2idx.get(w) < VOCAB_SIZE - 1 and word2idx.get(w) is not None else VOCAB_SIZE - 1  for w in p_rm.sub(' ',p_mark.sub(r' \1 ', jobads.get('mandatory_skill_keyword').encode('utf-8').lower())).split(' ') if w != ''])], maxlen=mandatory_skill_keyword_len, value=0)
  
  jobads_inx['num_features'] = jobads.get('num_features')
  
  
  #num_features =  comments_train_pandas.iloc[:,16:24].as_matrix()
  #16  CASE WHEN job_auto_forwarded_flag  = 'True' THEN 1 ELSE 0 AS job_auto_forwarded_flag,
  #17  CASE WHEN job_internship_flag  = 'True' THEN 1 ELSE 0 AS job_internship_flag,
  #18  CASE WHEN job_salary_visible  = 'True' THEN 1 ELSE 0 AS job_salary_visible,
  #19  CASE WHEN company_recruitment_firm_flag  = 'True' THEN 1 ELSE 0 AS company_recruitment_firm_flag,
  #20  coalesce(job_monthly_salary_min, 0 ) AS job_monthly_salary_min,
  #21  coalesce(job_monthly_salary_max, 0 ) AS job_monthly_salary_max,
  #22  coalesce(job_posting_date_start_datediff, 0 ) AS job_posting_date_start_datediff,
  #23  coalesce(job_posting_date_end_datediff, 0 ) AS job_posting_date_end_datediff
  #24  coalesce(years_of_experience, 0 ) AS years_of_experience,
  
  jobads_inx['id_apply'] = [jobads.get('id')]
  
  return jobads_inx



def spremodel_load(premodel_file):
    with open(premodel_file, 'rb') as f:
      premodel = pickle.load(f)
    return premodel


def smodel_load(model_file):
    model = load_model(model_file, custom_objects={'Attention': Attention})
    return model
                

def scoring(jobads_inx, model):
  
  preds = model.predict([jobads_inx['job_title'],jobads_inx['job_seniority_level'],jobads_inx['job_industry'], jobads_inx['job_description'], jobads_inx['job_requirement'], jobads_inx['job_employment_type'], jobads_inx['company_name'], jobads_inx['company_size'], jobads_inx['job_specializations'], jobads_inx['job_roles'], jobads_inx['job_work_locations'], jobads_inx['company_location'], jobads_inx['qualification_code'], jobads_inx['field_of_study'], jobads_inx['mandatory_skill_keyword'], jobads_inx['num_features']])
  
  results = {}
  
  results['reach'] = [preds[0], [np.exp(np.log(preds[0]) - np.sqrt(2)), np.exp(np.log(preds[0]) + np.sqrt(2))]]
  results['view'] =  [preds[1], [np.exp(np.log(preds[1]) - np.sqrt(0.7)), np.exp(np.log(preds[1]) + np.sqrt(0.7))]]
  results['application'] = [preds[2], [np.exp(np.log(preds[2]) - np.sqrt(0.9)), np.exp(np.log(preds[2]) + np.sqrt(0.9))]]

  return results

def handler(processed_payload):
    m_input = jobads2inx(processed_payload, current_app.pre_model)
    score_output = scoring(m_input, current_app.model)

    return score_output