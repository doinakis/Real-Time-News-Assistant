'''
@File    :   actions.py
@Time    :   2022/04/19 21:11:38
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   This file contains custom actions for the RASA bot. https://rasa.com/docs/rasa/custom-actions
'''
import os, sys
import threading
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))
from typing import Any, Text, Dict, List
from threading import Thread, Lock
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from qasystem.QASystem import *
from scrape.UHScrape import scrape_uh
from scrape.CNNScrape import scrape_cnn
from scrape.SPORT24Scrape import scrape_sport24
import logging

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)

_dbLock = threading.Lock()

class ActionAnswerQuestion(Action):
  '''
  The class that contains the bot answer question action
  '''
  def name(self) -> Text:
    return "bot_answer_question"

  def __init__(self) -> None:
    logger.debug(f'Conneting to the database.')
    self.db = Database()
    self.db.connect()
    logger.debug(f'Initializting QASystem.')
    self.qa = QASystem(database=self.db)

  def run(self, dispatcher: CollectingDispatcher,
          tracker: Tracker,
          domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

    question = tracker.latest_message["text"]
    with _dbLock:
      answer = self.qa.pipe.run(
        query=f'{question}', params={"ESRetriever": {"top_k": 3}, "Reader": {"top_k": 1}}
      )

    logger.info(answer)
    if answer['answers']:
      # TODO Optimize the probability threshold.
      dispatcher.utter_message(text=answer['answers'][0].answer)
    else:
      dispatcher.utter_message(text='Λυπάμαι αλλά δεν βρέθηκε καμία απάντηση για τη συγκεκριμένη ερώτηση :(')

    return []

def db_update(db):
  '''
  Update database on demand
  :param dispatcher: Class to provide messages to the user
  :param db: Database to write the scrapped documents to
  '''
  logger.info('Update of the database')

  articles = []
  articles_uh = []
  articles_cnn = []
  articles_sport24 = []
  uh_thread = Thread(target=scrape_uh, args=('./on-demand-db/', 1, articles_uh))
  cnn_thread = Thread(target=scrape_cnn, args=('./on-demand-db/', 1, 10, articles_cnn))
  sport24_thread = Thread(target=scrape_sport24, args=('./on-demand-db/', 1, 10, articles_sport24))

  uh_thread.start()
  cnn_thread.start()
  sport24_thread.start()

  uh_thread.join()
  cnn_thread.join()
  sport24_thread.join()

  articles = articles_uh + articles_cnn + articles_sport24
  with _dbLock:
    db.add_documents(articles)

  logger.info('Update of the database complete')


class ActionDatabaseUpdate(Action):
  '''
  The class that updates the database documents on demand
  '''
  def name(self) -> Text:
    return "bot_db_update"

  def __init__(self) -> None:
    logger.info('Conneting to the database.')
    self.db = Database()
    self.db.connect()
    self.db_update_thread = Thread(target=db_update, args=(self.db,))

  def run(self, dispatcher: CollectingDispatcher,
          tracker: Tracker,
          domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

    if not self.db_update_thread.is_alive():
      dispatcher.utter_message(text="Αναβάθμιση της βάσης στο παρασκήνιο.")
      self.db_update_thread.start()
    else:
      dispatcher.utter_message(text="Η βάση αναβαθμίζεται αυτή τη στιγμή.")

    return []