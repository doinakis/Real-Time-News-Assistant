'''
@File    :   db_to_single_file.py
@Time    :   2022/04/14 21:03:17
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   None
'''
import os, json
import pandas as pd


def scan_directory(path):
  '''
  Scan directory for json files
  :param path: Directory to scan
  '''
  db_dict = []
  with os.scandir(path) as it:
    for entry in it:
      if entry.name.endswith('.json') and entry.is_file() and entry.name != 'scrapped_links.json':
        data = json.load(open(entry.path))
        title = data['meta']['name']
        db_dict[f'{title}'] = data