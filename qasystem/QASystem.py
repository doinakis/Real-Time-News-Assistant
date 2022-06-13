'''
@File    :   QASystem.py
@Time    :   2022/04/10 13:54:12
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   This file contains classes to initialize a Database
             and a Question Answering System.
'''
from haystack.schema import Document
from haystack.pipelines import Pipeline
from haystack.nodes.base import BaseComponent
from haystack.nodes.connector import Crawler
from haystack.nodes.preprocessor.preprocessor import PreProcessor
from haystack.nodes import BM25Retriever, TfidfRetriever, FARMReader
from haystack.document_stores import ElasticsearchDocumentStore
import os, json
from classifier.Classifier import Classifier
from utils.utils import Singleton


class ClassificationNode(BaseComponent):
  '''
  The classification Node of the pipeline
  '''
  outgoing_edges = 1

  def __init__(self, classifier):
    self.classifier = classifier

  def run(self, query):
    '''
    :param query: The query to classify

    :returns:
        Touple with the dictionary with the classified query and the outgoing edge nubmer
    '''
    _, label = self.classifier.classify(query)
    if label is None:
      output = {
        "query": query
      }
    else:
      output = {
        "query": query,
        "filters": {"category": [f"{label}"]}
      }
    return (output, "output_1")


class MergeNode(BaseComponent):
  '''
  The merge Node merges multiple documents returned together. If multiple documents
  contain information about the same query there is a chance that the first document
  is not the one containing the answer to the query. In order to maximize the readers
  chance to find the correct answer we merge those documents into a single one and pass
  this to the reader.
  '''
  outgoing_edges = 1

  def run(self, documents):
    '''
    :param documents: The documents returned by the retriever

    :returns:
        Touple with the dictionary with the merged documents and the outgoing edge nubmer
    '''
    tmp_doc = []
    for document in documents:
      tmp_doc.append(document.content)
    document = Document(content = " ".join(tmp_doc), id ="0")

    # Handle no document returned
    if len(document.content) == 0:
      document.content = " "
    output = {
      "documents": [document]
    }

    return (output, "output_1")


class Database(metaclass=Singleton):
  '''
  The class that handles the connection to the document store.
  '''
  def __init__(self):
    self.processor = PreProcessor(clean_empty_lines=True,
                                  clean_whitespace=True,
                                  clean_header_footer=True,
                                  split_by="word",
                                  split_length=200,
                                  split_respect_sentence_boundary=True,
                                  language="el")
    self.crawler = Crawler(output_dir="crawled_files")

  def connect(self, host="localhost", port=9200, username="", password="", scheme="http",index="document", analyzer="greek"):
    '''
    Connect to a running database
    '''
    self.document_store = ElasticsearchDocumentStore(host=host,
                                                    port=port,
                                                    scheme=scheme,
                                                    username=username,
                                                    password=password,
                                                    index=index,
                                                    analyzer=analyzer)

  def add_documents(self, dicts):
    '''
    It runs the preprocess function on the documents and then adds it to the document store

    :param dicts: List of dictonaries containing the documents
    '''

    docs = self.processor.process(dicts)
    self.document_store.write_documents(docs)

  def add_documets_folder(self, folder_path):
    '''
    Add documents from folder

    :param folder_path: The path to the folder containing the documents
    '''
    dicts = []
    with os.scandir(folder_path) as folder:
      for file in folder:
        if(file.name.endswith(".json")) and file.is_file() and file.name != "scrapped_links.json":
          data = json.load(open(file.path))
          dicts.append(data)
    self.add_documents(dicts=dicts)

  def crawl_web(self, urls):
    '''
    Crawl a website and add it to the database

    :param urls: List of urls to crawl content from
    '''

    #TODO add custom crawlers
    dicts = self.crawler.run(url=urls, crawler_depth=0, overwrite_existing_files=True, return_documents=True)
    self.add_documents(dicts[0]['documents'])

  def delete_all_documents(self):
    '''
    Deletes all documents from the document store
    '''
    self.document_store.delete_documents()


class QASystem():
  '''
  The class containing the Question Answering System

  :param database: A database class
  :param classifier: A classifier class
  :param reader_model: String containing the name of a Huggingface model or the path to a local one
  :param max_seq_len: Maximum input to the model
  :param doc_stride: The doc stride used for searching
  :param use_gpu: Whether to use GPU for acceleration or not
  '''
  def __init__(self, database, classifier=None, reader_model="deepset/xlm-roberta-large-squad2", max_seq_len=256, doc_stride=128, use_gpu=True, retriever="bm25"):
    if not database:
      raise Exception("Database Object needed")

    self.db = database

    if retriever == "bm25":
      self.retriever = BM25Retriever(document_store=self.db.document_store)
    if retriever == "tfidf":
      self.retriever = TfidfRetriever(document_store=self.db.document_store)

    self.reader = FARMReader(model_name_or_path=reader_model, max_seq_len=max_seq_len, doc_stride=doc_stride, use_gpu=use_gpu, progress_bar=False)
    self.pipe = Pipeline()
    self.pipe.add_node(component=ClassificationNode(Classifier(classifier)), name="ClassificationNode", inputs=["Query"])
    self.pipe.add_node(component=self.retriever, name="Retriever", inputs=["ClassificationNode"])
    self.pipe.add_node(component=self.reader, name="Reader", inputs=["Retriever"])

  def pipeline(self, query, date, top_retriever=1, top_reader=1):
    '''
    Asks a question to the system
    :param query: The input question to the system
    :param date: The date of the oldest document to search
    :param top_retriever: The number of documents that the retriever will pass to the reader
    :param top_reader: The number of candidate answers the reader found

    :returns:
        prediction: Pipelines prediction
    '''
    filters = {}
    if date is not None:
      filters = {"date": {"$gte":date}}

    prediction = self.pipe.run(
      query=f'{query}', params={"Retriever": {"top_k": top_retriever, "filters": filters}, "Reader": {"top_k": top_reader}}
    )

    return prediction


if __name__== '__main__':
  print('Main function')