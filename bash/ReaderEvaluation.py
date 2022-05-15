'''
@File    :   ReaderEvaluation.py
@Time    :   2022/05/15 01:55:48
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Script to finetune Readers hyperparameters.
'''
import os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))
import argparse, logging
from datetime import datetime

from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

TIME = datetime.now().strftime('%H_%M_%S_%d_%m_%Y.log')
logging.basicConfig(filename=f'../logs/ReaderEvaluationLogs/{TIME}', filemode='w', format="%(levelname)s:%(name)s:%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level='INFO')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from haystack.schema import Document
from haystack.nodes import FARMReader
from qasystem.QASystem import *
from evaluation.xquadevaluation import *
import pandas as pd

logging.getLogger('WDM').disabled = True
logging.getLogger('haystack.nodes.connector.crawler').disabled = True
logging.getLogger('elasticsearch').setLevel(logging.WARNING)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='QASystem Evaluation script.')
  parser.add_argument('--xquad_file', required=True, help='path to the dataset')
  parser.add_argument('--model', required=True, help='Reader model to evalutate')
  args = parser.parse_args()

  reader = FARMReader(model_name_or_path=args.model, max_seq_len=256, doc_stride=128, use_gpu=True, progress_bar=False)
  df, _ = xquad_data_prepare(args.xquad_file)

  f1_scores = []
  em_scores = []

  for index, document in tqdm(df.iterrows(), total=df.shape[0], desc='Predicting'):
    tmp_doc = Document(content=document.content, id=document.doc_id)
    prediction = reader.predict(query=document.question, documents=[tmp_doc], top_k=1)

    em_scores.append(compute_em(prediction['answers'][0].answer, document.answer['text']))
    f1_scores.append(compute_f1(prediction['answers'][0].answer, document.answer['text']))

  scores = pd.DataFrame()
  scores['em'] = em_scores
  scores['f1'] = f1_scores


  logger.info('------------------------------')
  logger.info('Question Answering System info')
  logger.info(f'Reader: {args.model}')
  logger.info(f'Exact Match: {scores.em.mean()}')
  logger.info(f'F1-Score: {scores.f1.mean()}')
  logger.info('------------------------------')