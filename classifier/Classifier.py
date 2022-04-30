'''
@File    :   Classifier.py
@Time    :   2022/04/29 14:59:56
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Module to handle the two types of classification methods.
'''
import os, pickle
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertForSequenceClassification
from torch import device


class BertClassifier():
  '''
  Class for the bert classifier
  '''
  def __init__(self, classifier_path):
    self.tokenizer = BertTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
    self.model = BertForSequenceClassification.from_pretrained(classifier_path).to(device('cpu'))
    return

  def classify(self, query):
    inputs = self.tokenizer(query, return_tensors='pt')
    output = self.model(**inputs)
    prediction = output.logits.detach().cpu().numpy().argmax(axis=-1)[0]
    return prediction


class TFIDFClassifier():
  '''
  Class for the TFIDF classifier
  '''
  def __init__(self, classifier_path):
    with os.scandir(classifier_path) as it:
      for entry in it:
        if entry.name.endswith('.bin') and entry.is_file():
          vec_sel_struct = pickle.load(open(entry.path, 'rb'))
          self.vectorizer, self.selector = vec_sel_struct['vectorizer'], vec_sel_struct['selector']
        elif entry.name.endswith('.h5') and entry.is_file():
          self.model = load_model(entry.path)

  def classify(self, query):
    input_text = np.asarray(self.vectorizer.transform([f'{query}']).todense())
    input_vec = self.selector.transform(input_text).astype('float32')
    prediction = np.argmax(self.model.predict(input_vec), axis=-1)[0]
    return prediction


class Classifier():
  '''
  Class for the Generic Classifier class
  '''
  def __init__(self, classifier):
    absolute_path = '/home/doinakis/github/Real-Time-News-Assistant/classifier/models'
    if classifier == 'bert':
      self.classifier = BertClassifier(classifier_path=f'{absolute_path}/bert')
    elif classifier == 'tfidf':
      self.classifier = TFIDFClassifier(classifier_path=f'{absolute_path}/tfidf')
    else:
      self.classifier = None
    return

  '''
  Function that converts the prediction from a number to a label

  :param prediction: The predicted class (integer)
  '''
  def to_label(self, prediction):
    if prediction == 0:
      return 'sport'
    elif prediction == 1:
      return 'tech'
    elif prediction == 2:
      return 'gaming'
    elif prediction == 3:
      return 'movie'
    elif prediction == 4:
      return 'politiki'
    else:
      return 'None'

  '''
  Function that predicts the class of the query

  :param query: Query which class will be predicted
  '''
  def classify(self, query):
    if self.classifier is None:
      return None, None
    else:
      prediction = self.classifier.classify(query)
      return prediction, self.to_label(prediction)


