'''
@File    :   QAEvaluation.py
@Time    :   2022/04/10 19:44:21
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Script to evaluate the question answering system's performance
             Provides metrics about the retrivers Accuracy Score and the Accuracy
             and F1-Score for the QA Task. The readers Accuracy and F1-Score are
             affected and limited by the retrivers performance.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
from qasystem.QASystem import *
from evaluation.xquadevaluation import *
import argparse
from tqdm.contrib import tzip



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='QASystem Evaluation script.')
  parser.add_argument('--xquad_file', required=True, help='path to the dataset')
  parser.add_argument('--reader_model', required=True, help='reader model to be used for the system evaluation')
  parser.add_argument('--retriever_model', required=True, help='retriver model to be used. (currently support only for bm25 and tfidf)')

  args = parser.parse_args()

  db = Database()
  db.connect()
  db.delete_all_documents()

  df, dicts = xquad_data_prepare(args.xquad_file)
  db.add_documents(dicts=dicts)

  qa = QASystem(database=db, reader_model=args.reader_model)

  f1_scores = []
  em_scores_f1 = []
  em_scores = []
  retriever_em = []

  for question, answer, doc_id in tzip(df.question, df.answer, df.doc_id):
    prediction = qa.pipe.run(
      query=f"{question}", params={"ESRetriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    )

    retriever_em.append(int(prediction['documents'][0].meta['name'] == doc_id))
    em = compute_em(prediction['answers'][0].answer, answer['text'])
    em_scores.append(em)

    f1 = compute_f1(prediction['answers'][0].answer, answer['text'])
    f1_scores.append(f1)

    if f1 == 1:
      em_scores_f1.append(1)
    else:
      em_scores_f1.append(0)

  scores = pd.DataFrame()
  scores['em'] = em_scores
  scores['f1'] = f1_scores
  scores['em_f1'] = em_scores_f1
  scores['retriever_em'] = retriever_em

  print(f'Exact Match: {scores.em.mean()}')
  print(f'F1-Score: {scores.f1.mean()}')
  print(f'EM using F1: {scores.em_f1.mean()}')
  print(f'{args.retriever_model} Accuracy: {scores.retriever_em.mean()}')