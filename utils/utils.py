'''
@File    :   utils.py
@Time    :   2022/04/28 19:48:38
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Module that contains utilities definitions
'''
import threading
from tokenize import Single



class Singleton(type):
  '''
  Decorator class to handle singleton objects.
  '''
  _instances = {}
  _lock = threading.Lock()
  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      with cls._lock:
        if cls not in cls._instances:
          cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]
