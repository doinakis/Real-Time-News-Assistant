'''
@File    :   classifier-evaluation.py
@Time    :   2022/04/29 19:12:46
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Script to for classifier evaluation on actual dataset.
'''
import argparse, logging
import pandas as pd
from Classifier import *
logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='TFIDF classifier training script.')
  parser.add_argument('--dataset', required=True, help='path to the validation dataset')
  parser.add_argument('--classifier', required=True, help='classifier to evaluate. (currently supporting tfidf and bert)')
  args = parser.parse_args()

  classifier = Classifier(args.classifier)
  df = pd.read_csv(args.dataset)

  accuracy = 0
  queries = 0

  for category in df.columns:
    queries += len(df[category].dropna())
    for query in df[category].dropna():
      _, label = classifier.classify(query=query)
      if label == category:
        accuracy += 1

  accuracy = accuracy / queries

  logger.info(f'{args.classifier} Accuracy: {accuracy}')