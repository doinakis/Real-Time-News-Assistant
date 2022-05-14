'''
@File    :   RetrieverEvaluation.py
@Time    :   2022/05/14 21:45:39
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Script to finetune Retrievers hyperparameters.
'''
import os
import argparse, logging
from datetime import datetime

from numpy import append
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

TIME = datetime.now().strftime('%H_%M_%S_%d_%m_%Y.log')
logging.basicConfig(filename=f'./Logs/RetrieverEvaluationLogs/{TIME}', filemode='w', format="%(levelname)s:%(name)s:%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level='INFO')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from haystack.nodes import BM25Retriever, TfidfRetriever
from qasystem.QASystem import *
from evaluation.xquadevaluation import *
import pandas as pd
from tqdm.contrib import tzip

logging.getLogger('WDM').disabled = True
logging.getLogger('haystack.nodes.connector.crawler').disabled = True
logging.getLogger('elasticsearch').setLevel(logging.WARNING)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='QASystem Evaluation script.')
  parser.add_argument('--xquad_file', required=True, help='path to the dataset')
  parser.add_argument('--top_k_retriever', required=False, default=1, type=int, help='number of documents the retriever pass to the reader')

  args = parser.parse_args()

  db = Database()
  db.connect()
  db.delete_all_documents()

  df, dicts = xquad_data_prepare(args.xquad_file)
  db.add_documents(dicts=dicts)

  bm25 = BM25Retriever(db.document_store)
  tfidf = TfidfRetriever(db.document_store)

  bm25_acc = []
  tfidf_acc = []

  for question, answer, doc_id in tzip(df.question, df.answer, df.doc_id):
    bm25_docs = bm25.retrieve(query=question, top_k=args.top_k_retriever)
    tfidf_docs = tfidf.retrieve(query=question, top_k=args.top_k_retriever)

    bm25_file_names = []
    tfidf_file_names = []
    for bm25_doc, tfidf_doc in zip(bm25_docs, tfidf_docs):
      bm25_file_names.append(bm25_doc.meta['name'])
      tfidf_file_names.append(tfidf_doc.meta['name'])

    bm25_acc.append(int(doc_id in bm25_file_names))
    tfidf_acc.append(int(doc_id in tfidf_file_names))

  acc = pd.DataFrame()
  acc['bm25'] = bm25_acc
  acc['tfidf'] = tfidf_acc
  logger.info('------------------------------')
  logger.info(f'Retriver Evaluation info top_k:{args.top_k_retriever}')
  logger.info(f'BM25: {acc.bm25.mean()}')
  logger.info(f'TFIDF: {acc.tfidf.mean()}')
  logger.info('------------------------------')