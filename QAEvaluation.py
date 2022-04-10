'''
@File    :   QAEvaluation.py
@Time    :   2022/04/10 19:44:21
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   None
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
  args = parser.parse_args()
  db = Database()
  db.connect()
  db.delete_all_documents()
  # xquad_path = "/home/doinakis/github/haystack/xquad-dataset/xquad.el.json"
  df, dicts = xquad_data_prepare(args.xquad_file)

  db.add_documents(dicts=dicts)

  qa = QASystem(database=db)

  f1_scores = []
  em_scores = []
  for question, answer in tzip(df.question, df.answer):
    prediction = qa.pipe.run(
      query=f"{question}", params={"ESRetriever": {"top_k": 1}, "Reader": {"top_k": 1}}
    )

    em = compute_em(prediction['answers'][0].answer, answer['text'])
    print(em)
    if em == 1:
      f1_scores.append(1)
    else:
      f1_scores.append(compute_f1(prediction['answers'][0].answer, answer['text']))

  scores = pd.DataFrame()
  scores['em'] = em_scores
  scores['f1'] = f1_scores

  print(scores.em.mean())
  print(scores.f1.mean())