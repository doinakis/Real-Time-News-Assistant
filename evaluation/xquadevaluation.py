'''
@File    :   xquadevaluation.py
@Time    :   2022/04/10 17:18:02
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Module that prepares xquad files for evaluation with the system.
'''
import os, json, string
import pandas as pd
import unicodedata

def xquad_data_prepare(data_path):
  '''
  Prepare the data for the system evaluation
  :param data: Dictionary with SQuAD v1.1 based format

  :returns:
      df: Pandas dataframe with the contexts questions and answers
      dicts: List of dictionaries with the unique contexts
  '''
  squad = json.load(open(data_path))

  if squad['version'] != '1.1':
    raise Exception("Provide a dataset based on SQuad v1.1")

  df = pd.DataFrame()
  dicts = []
  contexts = []
  questions = []
  answers = []
  document_ids =[]
  id = 0
  for data in squad['data']:
    for paragraph in data['paragraphs']:
      context  = paragraph['context']
      document_id = f'document{id}.json'
      id += 1
      for qa in paragraph['qas']:
        question = qa['question']
        for answer in qa['answers']:
          contexts.append(context)
          questions.append(question)
          answers.append(answer)
          document_ids.append(document_id)

  df['content'] = contexts
  df['question'] = questions
  df['answer'] = answers
  df['doc_id'] = document_ids
  id = 0
  for content in df.content.unique():
    tmp_dict = {
      "content": content,
      "meta": {
        "name": f'document{id}.json'
      }
    }
    id += 1
    dicts.append(tmp_dict)

  return df, dicts


def strip_accents_and_lowercase(text):
  '''
  Strip accents and meke the text lowercase
  :param text: String to strip accents and make lowecase
  '''
  return ''.join(c for c in unicodedata.normalize('NFD', text)
                  if unicodedata.category(c) != 'Mn').lower()


def compute_em(prediction, truth):
  '''
  Compare prediction and truth strings
  :param prediction: The predicted string
  :param truth: The actual string

  :returns:
    Whether the two strings are exactly matched
  '''
  return int(strip_accents_and_lowercase(prediction.replace(" ", "")) == strip_accents_and_lowercase(truth.replace(" ", "")))


def compute_f1(prediction, truth):
  '''
  Compute f1 score for the input strings
  :param prediction: The predicted string
  :param truth: The actual string

  :returns:
    F1: The f1 score between 2 strings
  '''
  if compute_em(prediction,truth) == 1:
    return 1

  pred_tokens = strip_accents_and_lowercase(prediction).split()
  truth_tokens = strip_accents_and_lowercase(truth).split()

  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)

  common_tokens = set(pred_tokens) & set(truth_tokens)

  if len(common_tokens) == 0:
    return 0

  precision = len(common_tokens) / len(pred_tokens)
  recall = len(common_tokens) / len(truth_tokens)

  return 2 * (precision * recall) / (precision + recall)


# def remove_white_space(text):
#   '''
#   Remove white spaces from text
#   :param text: String to remove white spaces from
#   '''
#   return " ".join(text.split())

# def remove_punctuation(text):
#   '''
#   Remove punctuation from input text
#   :param text: String to remover punctuation from
#   '''
#   exclude = set(string.punctuation)
#   return "".join(ch for ch in text if ch not in exclude)

# def lower_case(text):
#   '''
#   Convert capital letters to lower case
#   :param text: String to convert to lowercase
#   '''
#   return text.lower()

# def normalize(text):
#   '''
#   Apply normalization of the text
#   :param text: String to apply the normalization to
#   '''
#   return remove_white_space(remove_punctuation(lower_case(text)))