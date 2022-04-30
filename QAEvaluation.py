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
from asyncio.log import logger
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
from qasystem.QASystem import *
from evaluation.xquadevaluation import *
from classifier.Classifier import *
import argparse, logging
from datetime import datetime
from tqdm.contrib import tzip

TIME = datetime.now().strftime('%H_%M_%d_%m_%Y.log')
logging.basicConfig(filename=f'./QAEvaluationLogs/{TIME}',format="%(levelname)s:%(name)s:%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level='INFO')
logging.getLogger('WDM').disabled = True
logging.getLogger('haystack.nodes.connector.crawler').disabled = True
logging.getLogger('elasticsearch').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='QASystem Evaluation script.')
  parser.add_argument('--xquad_file', required=True, help='path to the dataset')
  parser.add_argument('--classifier', required=False, default=None, help='what type of query classification to use. (currently supporting tfidf and Bert')
  parser.add_argument('--reader_model', required=False, default='deepset/xlm-roberta-large-squad2', help='reader model to be used for the system evaluation')
  parser.add_argument('--retriever_model', required=False, default='bm25', help='retriever model to be used. (currently supporting bm25 and tfidf)')
  parser.add_argument('--top_k_retriever', required=False, default=1, type=int, help='number of documents the retriever pass to the reader')
  parser.add_argument('--top_k_reader', required=False, default=1, type=int, help='number of best predicted answers returned from the reader')

  args = parser.parse_args()

  db = Database()
  db.connect()
  db.delete_all_documents()

  df, dicts = xquad_data_prepare(args.xquad_file)
  db.add_documents(dicts=dicts)

  qa = QASystem(database=db, classifier=Classifier(args.classifier),reader_model=args.reader_model, retriever=args.retriever_model)

  f1_scores = []
  em_scores = []
  retriever_em = []

  for question, answer, doc_id in tzip(df.question, df.answer, df.doc_id):
    prediction = qa.pipe.run(
      query=f'{question}', params={"ESRetriever": {"top_k": args.top_k_retriever}, "Reader": {"top_k": args.top_k_reader}}
    )

    retriever_em.append(int(prediction['documents'][0].meta['name'] == doc_id))
    em_scores.append(compute_em(prediction['answers'][0].answer, answer['text']))
    f1_scores.append(compute_f1(prediction['answers'][0].answer, answer['text']))

  scores = pd.DataFrame()
  scores['em'] = em_scores
  scores['f1'] = f1_scores
  scores['retriever_em'] = retriever_em

  logger.info('------------------------------')
  logger.info('Question Answering System info')
  logger.info(f'Classifier: {args.classifier}')
  logger.info(f'Reader: {args.reader_model} top_k: {args.top_k_reader}')
  logger.info(f'Retriever: {args.retriever_model} top_k: {args.top_k_retriever} Accuracy: {scores.retriever_em.mean()}')
  logger.info(f'Exact Match: {scores.em.mean()}')
  logger.info(f'F1-Score: {scores.f1.mean()}')
  logger.info('------------------------------')
