'''
@File    :   train_tfidf.py
@Time    :   2022/04/29 16:22:35
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   This script is obtained from https://developers.google.com/machine-learning/guides/text-classification
             and adjusted to meet this project's needs.
'''
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.optimizer_v2.adam import Adam
import pickle, argparse, logging, json, random
import explore_data
import pandas as pd
import numpy as np

# logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)

# Vectorization parameters
# Range (inclusive) of n-gram sizes for tokenizing text.
NGRAM_RANGE = (1, 2)

# Limit on the number of features. We use the top 20K features.
TOP_K = 20000

# Whether text should be split into word or character n-grams.
# One of 'word', 'char'.
TOKEN_MODE = 'word'

# Minimum document/corpus frequency below which a token will be discarded.
MIN_DOCUMENT_FREQUENCY = 2

def ngram_vectorize(model_name, train_texts, train_labels, val_texts):
  """Vectorizes texts as n-gram vectors.

  1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

  # Arguments
      train_texts: list, training text strings.
      train_labels: np.ndarray, training labels.
      val_texts: list, validation text strings.

  # Returns
      x_train, x_val: vectorized training and validation texts
  """
  # Create keyword arguments to pass to the 'tf-idf' vectorizer.
  kwargs = {
          'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
          # 'dtype': 'int32', # commented to suppress warning
          'strip_accents': 'unicode',
          'decode_error': 'replace',
          'analyzer': TOKEN_MODE,  # Split text into word tokens.
          'min_df': MIN_DOCUMENT_FREQUENCY,
  }
  vectorizer = TfidfVectorizer(**kwargs)

  # Learn vocabulary from training texts and vectorize training texts.
  x_train = np.asarray(vectorizer.fit_transform(train_texts).todense())

  # Vectorize validation texts.
  x_val = np.asarray(vectorizer.transform(val_texts).todense())

  # Select top 'k' of the vectorized features.
  selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
  selector.fit(x_train, train_labels)

  # Save the vectorizer-selector pair
  with open(f"{model_name}-vec-sec.bin", 'wb') as file:
      pickle.dump({'vectorizer': vectorizer, 'selector': selector}, file)
  x_train = selector.transform(x_train).astype('float32')
  x_val = selector.transform(x_val).astype('float32')
  return x_train, x_val


def _get_last_layer_units_and_activation(num_classes):
  """Gets the # units and activation function for the last network layer.

  # Arguments
      num_classes: int, number of classes.

  # Returns
      units, activation values.
  """
  if num_classes == 2:
      activation = 'sigmoid'
      units = 1
  else:
      activation = 'softmax'
      units = num_classes
  return units, activation


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
  """Creates an instance of a multi-layer perceptron model.

  # Arguments
      layers: int, number of `Dense` layers in the model.
      units: int, output dimension of the layers.
      dropout_rate: float, percentage of input to drop at Dropout layers.
      input_shape: tuple, shape of input to the model.
      num_classes: int, number of output classes.

  # Returns
      An MLP model instance.
  """
  op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
  model = models.Sequential()
  model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

  for _ in range(layers-1):
      model.add(Dense(units=units, activation='relu'))
      model.add(Dropout(rate=dropout_rate))

  model.add(Dense(units=op_units, activation=op_activation))
  return model

def train_ngram_model(model_name,
                      data,
                      learning_rate=1e-3,
                      epochs=1000,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
  """Trains n-gram model on the given dataset.

  # Arguments
      data: tuples of training and test texts and labels.
      learning_rate: float, learning rate for training model.
      epochs: int, number of epochs.
      batch_size: int, number of samples per batch.
      layers: int, number of `Dense` layers in the model.
      units: int, output dimension of Dense layers in the model.
      dropout_rate: float: percentage of input to drop at Dropout layers.

  # Raises
      ValueError: If validation data has label values which were not seen
          in the training data.
  """
  # Get the data.
  (train_texts, train_labels), (val_texts, val_labels) = data

  # Verify that validation labels are in the same range as training labels.
  num_classes = explore_data.get_num_classes(train_labels)
  unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
  if len(unexpected_labels):
      raise ValueError('Unexpected label values found in the validation set:'
                        ' {unexpected_labels}. Please make sure that the '
                        'labels in the validation set are in the same range '
                        'as training labels.'.format(
                            unexpected_labels=unexpected_labels))

  # Vectorize texts.
  x_train, x_val = ngram_vectorize(model_name, train_texts, train_labels, val_texts)

  # Create model instance.
  model = mlp_model(layers=layers,
                    units=units,
                    dropout_rate=dropout_rate,
                    input_shape=x_train.shape[1:],
                    num_classes=num_classes)

  # Compile model with learning parameters.
  if num_classes == 2:
      loss = 'binary_crossentropy'
  else:
      loss = 'sparse_categorical_crossentropy'
  optimizer = Adam(learning_rate=learning_rate)
  model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

  # Create callback for early stopping on validation loss. If the loss does
  # not decrease in two consecutive tries, stop training.
  callbacks = [tf.keras.callbacks.EarlyStopping(
      monitor='val_loss', patience=2)]

  # Train and validate model.
  history = model.fit(
          x_train,
          train_labels,
          epochs=epochs,
          callbacks=callbacks,
          validation_data=(x_val, val_labels),
          verbose=2,  # Logs once per epoch.
          batch_size=batch_size)

  # Print results.
  history = history.history
  logger.info('Validation accuracy: {acc}, loss: {loss}'.format(
          acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

  # Save model.
  model.save(f"{model_name}.h5")
  return model, history['val_acc'][-1], history['val_loss'][-1]

def model_evaluation(model, vec_sec_struct, test_data):
  (test_texts, test_labels) = test_data
  vectorizer, selector = vec_sec_struct['vectorizer'], vec_sec_struct['selector']
  input_text = np.asarray(vectorizer.transform(test_texts).todense())
  input_vec = selector.transform(input_text).astype('float32')
  prediction = np.argmax(model.predict(input_vec), axis=-1)
  acc = np.equal(prediction, test_labels)
  acc = sum(acc) / len(acc)
  return acc


def read_from_database(db_path):
  with open(f'{db_path}/train_dataset.json', 'r') as file:
    data = pd.read_json(file)

  with open(f'{db_path}/val_dataset.json', 'r') as file:
    val_data = pd.read_json(file)

  with open(f'{db_path}/test_dataset.json', 'r') as file:
    test_data = pd.read_json(file)

  for index, entry in data.iterrows():
    if entry.category == 'football' or entry.category == 'tennis' or entry.category == 'basket':
      entry.category = 'sport'
  for index, entry in val_data.iterrows():
    if entry.category == 'football' or entry.category == 'tennis' or entry.category == 'basket':
      entry.category = 'sport'

  for index, entry in test_data.iterrows():
    if entry.category == 'football' or entry.category == 'tennis' or entry.category == 'basket':
      entry.category = 'sport'

  data = data.sample(frac=0.5,random_state=200) # Due to hardware limitations

  data = data.sample(frac=1).reset_index(drop=True)
  val_data = val_data.sample(frac=1).reset_index(drop=True)
  test_data = test_data.sample(frac=1).reset_index(drop=True)

  train_texts = data.content.to_list()
  val_texts = val_data.content.to_list()
  test_texts = test_data.content.to_list()

  conditions =      [data["category"]=="sport",  \
                     data["category"]=="tech",   \
                     data["category"]=="gaming", \
                     data["category"]=="movies", \
                     data["category"]=="politiki"]
  val_conditions =  [val_data["category"]=="sport",  \
                     val_data["category"]=="tech",   \
                     val_data["category"]=="gaming", \
                     val_data["category"]=="movies", \
                     val_data["category"]=="politiki"]
  test_conditions = [test_data["category"]=="sport",  \
                     test_data["category"]=="tech",   \
                     test_data["category"]=="gaming", \
                     test_data["category"]=="movies", \
                     test_data["category"]=="politiki"]
  choices     = [0, 1, 2, 3, 4]
  train_labels = np.select(conditions, choices, default=np.nan).astype(int)
  val_labels = np.select(val_conditions, choices, default=np.nan).astype(int)
  test_labels = np.select(test_conditions, choices, default=np.nan).astype(int)

  data = ((train_texts,train_labels), (val_texts, val_labels))
  test_data = (test_texts, test_labels)

  return data, test_data


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='TFIDF classifier training script.')
  parser.add_argument('--model_name', required=True, help='model desired name')
  parser.add_argument('--db_path', required= True, help='path to the stored documents')
  args = parser.parse_args()

  logger.info('Preprocess documents.')
  data, test_data = read_from_database(db_path=args.db_path)

  logger.info('Train the tfidf model.')
  model_path = f'/home/doinakis/github/Real-Time-News-Assistant/classifier/models/tfidf/{args.model_name}'
  model, val_acc, val_loss = train_ngram_model(model_path,
                                              data,
                                              learning_rate=1e-3,
                                              epochs=4,
                                              batch_size=4,
                                              layers=2,
                                              units=64,
                                              dropout_rate=0.2)

  logger.info('Training complete.')
  logger.info('Starting evaluation.')
  with open(f"{model_path}-vec-sec.bin", 'rb') as file:
    vec_sec_struct = pickle.load(file)

  accuracy = model_evaluation(model=model, vec_sec_struct=vec_sec_struct, test_data=test_data)
  logger.info(f'Accuracy: {accuracy}')