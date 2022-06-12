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

  y_true = None
  y_pred = None
  target_names = ['sport' , 'tech', 'gaming', 'movies', 'politiki', 'other']
  L2I = {label: i for i, label in enumerate(target_names)}

  for category in df.columns:
    queries += len(df[category].dropna())
    for query in df[category].dropna():
      logits, label = classifier.classify(query=query)

      if y_true is None:
        y_true = L2I[label].detach().cpu().numpy()
        y_pred = logits.detach().cpu().numpy()
      else:
        y_true = np.append(y_true, L2I[label].detach().cpu().numpy(), axis=0)
        y_pred = np.append(y_pred, logits.detach().cpu().numpy(), axis=0)

    y_pred = y_pred.argmax(axis=-1)

  accuracy = accuracy / queries

  logger.info(f'{args.classifier} Accuracy: {accuracy}')