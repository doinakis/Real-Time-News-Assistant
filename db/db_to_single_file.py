'''
@File    :   db_to_single_file.py
@Time    :   2022/04/14 21:03:17
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   (C)Copyright 2022 Michalis Doinakis
@Desc    :   Convert database to a single file and export train validation and test dataset.(70%, 20%, 10%)
'''
import os, json, argparse
import pandas as pd


def db_convert(db_path, single_file_path):
  '''
  Converts db with the following structure to a single file
  -db/
    -cnn/
      -politiki/
    -sport24/
      -basket/
      -football/
      -tennis/
    -uh/
      -gaming/
      -movies/
      -tennis

  :param db_path: Path to the db
  :param single_file_path: Location to store the single file database
  '''
  df = pd.DataFrame()
  db_dict = {}
  index = 0

  with os.scandir(f'{db_path}/uh/gaming') as it:
    for entry in it:
      if entry.name.endswith('.json') and entry.is_file() and entry.name != 'scrapped_links.json':
        data = json.load(open(entry.path))
        db_dict[f'article{index}'] = data
        index += 1
  with os.scandir(f'{db_path}/uh/movies') as it:
    for entry in it:
      if entry.name.endswith('.json') and entry.is_file() and entry.name != 'scrapped_links.json':
        data = json.load(open(entry.path))
        db_dict[f'article{index}'] = data
        index += 1
  with os.scandir(f'{db_path}/uh/tech') as it:
    for entry in it:
      if entry.name.endswith('.json') and entry.is_file() and entry.name != 'scrapped_links.json':
        data = json.load(open(entry.path))
        db_dict[f'article{index}'] = data
        index += 1
  with os.scandir(f'{db_path}/sport24/basket') as it:
    for entry in it:
      if entry.name.endswith('.json') and entry.is_file() and entry.name != 'scrapped_links.json':
        data = json.load(open(entry.path))
        data["meta"]["name"] = data["meta"]["url"].split("/")[-1] # This line was added due to a bug during scrapping
        db_dict[f'article{index}'] = data
        index += 1
  with os.scandir(f'{db_path}/sport24/football') as it:
    for entry in it:
      if entry.name.endswith('.json') and entry.is_file() and entry.name != 'scrapped_links.json':
        data = json.load(open(entry.path))
        data["meta"]["name"] = data["meta"]["url"].split("/")[-1] # This line was added due to a bug during scrapping
        db_dict[f'article{index}'] = data
        index += 1
  with os.scandir(f'{db_path}/sport24/tennis') as it:
    for entry in it:
      if entry.name.endswith('.json') and entry.is_file() and entry.name != 'scrapped_links.json':
        data = json.load(open(entry.path))
        data["meta"]["name"] = data["meta"]["url"].split("/")[-1] # This line was added due to a bug during scrapping
        db_dict[f'article{index}'] = data
        index += 1
  with os.scandir(f'{db_path}/cnn/politiki') as it:
    for entry in it:
      if entry.name.endswith('.json') and entry.is_file() and entry.name != 'scrapped_links.json':
        data = json.load(open(entry.path))
        db_dict[f'article{index}'] = data
        index += 1

  df = pd.DataFrame.from_dict(db_dict,orient='index')
  dataframe = pd.DataFrame()
  content = df.content
  meta = pd.json_normalize(df.meta)
  dataframe["content"] = content.tolist()
  dataframe = dataframe.join(meta)
  duplicate = dataframe[dataframe.content.duplicated()]
  dataframe = dataframe.drop(duplicate.index)           # Fix bug some files were in the database twice
  dataframe.to_json(f'{single_file_path}/single_file_db.json')


def db_split(db_single_file, other, to_store):
  '''
  Splits the single file db to training and test dataset

  :param db_single_file: Path to the single database file
  :param other: Path to data that has no label
  :param to_store: Path to folder to store the training and validation dataset.
  '''

  with open(f'{db_single_file}', 'r') as file:
    data = pd.read_json(file)

  with open(f'{other}', 'r') as file:
    other_data = pd.read_json(file)

  for index, entry in data.iterrows():
    if entry.category == 'football' or entry.category == 'tennis' or entry.category == 'basket':
      entry.category = 'sport'

  sport_df = data[data.category == 'sport']
  gaming_df = data[data.category == 'gaming']
  tech_df = data[data.category == 'tech']
  movies_df = data[data.category == 'movies']
  politiki_df = data[data.category == 'politiki']

  sport_train = sport_df.sample(frac=0.7,random_state=200)
  temp = sport_df.drop(sport_train.index)
  sport_test = temp.sample(frac=0.4, random_state=200)
  sport_val = temp.drop(sport_test.index)

  gaming_train =  gaming_df.sample(frac=0.7,random_state=200)
  temp = gaming_df.drop(gaming_train.index)
  gaming_test = temp.sample(frac=0.4, random_state=200)
  gaming_val = temp.drop(gaming_test.index)

  tech_train = tech_df.sample(frac=0.7,random_state=200)
  temp = tech_df.drop(tech_train.index)
  tech_test = temp.sample(frac=0.4, random_state=200)
  tech_val = temp.drop(tech_test.index)

  movies_train = movies_df.sample(frac=0.7,random_state=200)
  temp = movies_df.drop(movies_train.index)
  movies_test = temp.sample(frac=0.4, random_state=200)
  movies_val = temp.drop(movies_test.index)

  politiki_train = politiki_df.sample(frac=0.7,random_state=200)
  temp = politiki_df.drop(politiki_train.index)
  politiki_test = temp.sample(frac=0.4, random_state=200)
  politiki_val = temp.drop(politiki_test.index)

  other_train = other_data.sample(frac=0.7,random_state=200)
  temp = other_data.drop(other_train.index)
  other_test = temp.sample(frac=0.4, random_state=200)
  other_val = temp.drop(other_test.index)

  train_df = pd.concat([sport_train, gaming_train, tech_train, movies_train, politiki_train, other_train], ignore_index=True)
  test_df = pd.concat([sport_test, gaming_test, tech_test, movies_test, politiki_test, other_test], ignore_index=True)
  val_df = pd.concat([sport_val, gaming_val, tech_val, movies_val, politiki_val, other_val], ignore_index=True)

  train_df.to_json(f'{to_store}/train_dataset.json')
  test_df.to_json(f'{to_store}/test_dataset.json')
  val_df.to_json(f'{to_store}/val_dataset.json')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='TFIDF classifier training script.')
  parser.add_argument('--db_path', required=True, help='path to the db')
  parser.add_argument('--store_path', required= True, help='path to store the single file db')
  args = parser.parse_args()

  db_convert(args.db_path, args.store_path)
  db_split(f'{args.store_path}/single_file_db.json',
          f'{args.db_path}/other.json',
          args.store_path)