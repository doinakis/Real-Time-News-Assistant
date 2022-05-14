'''
@File    :   train_bert.py
@Time    :   2022/04/29 18:13:44
@Author  :   Michalis Doinakis
@Version :   0.0.1
@Contact :   doinakis.michalis@gmail.com
@License :   https://github.com/andreasgoulas/greek-bert-distil
@Desc    :   This file was obtained from https://github.com/andreasgoulas/greek-bert-distil
'''
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import argparse, logging
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import f1_score, classification_report

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

class Example:
    def __init__(self, sent0, sent1, label_id):
        self.sent0 = sent0
        self.sent1 = sent1
        self.label_id = label_id

class Features:
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def convert_example(example, tokenizer, max_seq_len):
    out = tokenizer(example.sent0, example.sent1, padding='max_length',
        max_length=max_seq_len, truncation=True)
    return Features(out['input_ids'], out['attention_mask'], out['token_type_ids'],
        example.label_id)

def get_tensor_dataset(features):
    input_ids = torch.tensor([x.input_ids for x in features], dtype=torch.long)
    input_masks = torch.tensor([x.input_mask for x in features], dtype=torch.bool)
    segment_ids = torch.tensor([x.segment_ids for x in features], dtype=torch.int)
    label_ids = torch.tensor([x.label_id for x in features], dtype=torch.long)
    return TensorDataset(input_ids, input_masks, segment_ids, label_ids)



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Bert classifier training script.')
  parser.add_argument('--train_dataset', required=True, help='path to the training dataset')
  parser.add_argument('--test_dataset', required=True, help='path to the test dataset')
  parser.add_argument('--store_path', required= True, help='path to store the single file db')
  args = parser.parse_args()

  with open(args.train_dataset, 'r') as file:
    data = pd.read_json(file)

  data = data.sample(frac=1).reset_index(drop=True)

  epochs=3
  tokenizer_max_len=125
  batch_size=32
  learning_rate=5e-05
  device = torch.device('cpu')

  labels = ['sport' , 'tech', 'gaming', 'movies', 'politiki', 'other']
  L2I = {label: i for i, label in enumerate(labels)}

  tokenizer = BertTokenizer.from_pretrained('nlpaueb/bert-base-greek-uncased-v1')
  model = BertForSequenceClassification.from_pretrained('nlpaueb/bert-base-greek-uncased-v1', num_labels=6).to(device)

  features = []
  for index, entry in tqdm(data.iterrows(), total=data.shape[0], desc='loading data'):
    example = Example(entry['content'], None, L2I[entry['category']])
    features.append(convert_example(example, tokenizer, tokenizer_max_len))

  train_dataset = get_tensor_dataset(features)
  optimizer = AdamW(model.parameters(), lr=learning_rate)
  train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

  for epoch in range(epochs):
    avg_loss = 0
    model.train()
    progress = tqdm(train_loader, desc='finetuning')
    for step, batch in enumerate(progress):
      batch = tuple(x.to(device) for x in batch)
      input_ids, input_masks, segment_ids, label_ids = batch
      optimizer.zero_grad()

      output = model(input_ids, attention_mask=input_masks, token_type_ids=segment_ids, labels=label_ids)

      loss = output.loss
      loss.backward()
      optimizer.step()

      avg_loss += loss.item()
      progress.set_postfix({'loss': loss.item()})

    avg_loss /= len(train_loader)
    logger.info(f'[epoch {epoch + 1}] loss = {avg_loss}')

  model.save_pretrained(args.store_path)

  with open(args.test_dataset, 'r') as file:
    test_data = pd.read_json(file)

  test_features = []

  for index, entry in tqdm(test_data.iterrows(), total=test_data.shape[0], desc='loading data'):
    example = Example(entry['content'], None, L2I[entry['category']])
    test_features.append(convert_example(example, tokenizer, tokenizer_max_len))

  test_dataset = get_tensor_dataset(test_features)
  target_names = labels

  test_loader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset),batch_size=batch_size)
  y_true = None
  y_pred = None

  model.eval()
  with torch.no_grad():
    for batch in test_loader:
      batch = tuple(x.to(device) for x in batch)
      input_ids, input_masks, segment_ids, label_ids = batch
      logits = model(input_ids, attention_mask=input_masks,
                  token_type_ids=segment_ids)[0]
      if y_true is None:
        y_true = label_ids.detach().cpu().numpy()
        y_pred = logits.detach().cpu().numpy()
      else:
        y_true = np.append(y_true, label_ids.detach().cpu().numpy(), axis=0)
        y_pred = np.append(y_pred, logits.detach().cpu().numpy(), axis=0)
  y_pred = y_pred.argmax(axis=-1)
  if target_names is not None:
    print(classification_report(y_true, y_pred, digits=3, target_names=target_names))
  f1 = f1_score(y_true, y_pred, average='macro')
  logger.info(f1)