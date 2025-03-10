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
from datetime import datetime
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))
from typing import Any, Text, Dict, List
from threading import Thread, Lock
import logging

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p", level='Debug')
logger = logging.getLogger(__name__)

from rasa_sdk import Action, Tracker
from rasa_sdk.forms import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from qasystem.QASystem import *
from scrape.UHScrape import scrape_uh
from scrape.CNNScrape import scrape_cnn
from scrape.SPORT24Scrape import scrape_sport24


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
    date = None
    now = datetime.now()
    if tracker.get_slot('time_slot') is not None and isinstance(tracker.get_slot('time_slot'), str):
      date = tracker.get_slot('time_slot').split('T')[0] # Get only the date (%y%m%d)

    question = tracker.latest_message["text"]

    if date is not None:
      question = question.replace(date, "") # Remove the date provided by the User (Maybe bug check it out)
      if (datetime.strptime(date, '%Y-%m-%d') > now):
        date = None

    print(question)

    with _dbLock:
      answer = self.qa.pipeline(query=f'{question}', date=date, top_retriever=3, top_reader=1)

    if answer['answers']:
      # TODO Optimize the probability threshold.
      ans = answer['answers'][0].answer
      url_spaced_ans = ans.replace(' ', '%20')
      link = answer['answers'][0].meta['url'] + f'#:~:text={url_spaced_ans}'
      text=f'Η απάντηση στην ερώτηση σου είναι: {ans}, σύμφωνα με [αυτό]({link}) το άρθρο.'
      dispatcher.utter_message(text=text)
    else:
      dispatcher.utter_message(text='Λυπάμαι αλλά δεν βρέθηκε καμία απάντηση για τη συγκεκριμένη ερώτηση :(')

    return [SlotSet("time_slot", None)]

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