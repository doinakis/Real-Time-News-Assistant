'''
@File    :   actions.py
@Time    :   2022/04/19 21:11:38
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   This file contains custom actions for the RASA bot. https://rasa.com/docs/rasa/custom-actions
'''
import logging, os, sys
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..')))

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from qasystem.QASystem import *

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)


class ActionAnswerQuestion(Action):
  '''
  The class that contains the bot answer question action
  '''
  def name(self) -> Text:
    return "bot_answer_question"

  def __init__(self) -> None:
    logger.debug(f"Conneting to the database.")
    self.db = Database()
    self.db.connect()
    logger.debug(f"Initializting QASystem.")
    self.qa = QASystem(database=self.db)

  def run(self, dispatcher: CollectingDispatcher,
          tracker: Tracker,
          domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

    question = tracker.latest_message["text"]
    answer = self.qa.pipe.run(
      query=f"{question}", params={"ESRetriever": {"top_k": 3}, "Reader": {"top_k": 1}}
    )

    # TODO add support for no answer found. Optimize the probability threshold.
    dispatcher.utter_message(text=answer['answers'][0].answer)

    return []
