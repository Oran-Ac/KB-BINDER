
'''dataset '''
from absl import flags
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from dataclasses import dataclass
import json
FLAGS = flags.FLAGS

def prepare_data(data_dir,dataset,status):
    data = []
    raw_data_path = os.path.join(FLAGS.data_root_path,dataset)
    if dataset == 'webqsp_0107':
        raw_data_path = raw_data_path +f'.{status}.json'
        raw_data = json.load(open(raw_data_path))
        for item in raw_data:
            data.append({
                'question':item['question'],
                'sparql_query':item['sparql_query']
            })
        # save the results
        os.makedirs(os.path.join(data_dir,dataset),exist_ok=True)
        file_path = os.path.join(data_dir,dataset,status+'.jsonl')
        with open(file_path,'w') as f:
            for item in data:
                f.write(json.dumps(item)+'\n')
    else:
        raise
    return data


def load_data(data_dir,dataset,status):
    data = []
    file_path = os.path.join(data_dir,dataset,status+'.jsonl')
    if not os.path.exists(file_path):
        return prepare_data(data_dir,dataset,status)
    with open(file_path,'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def prepare_data_sentence_only(row_data,tokenizer,max_length):
    prompt_input = [
       tokenizer.bos_token +  ' Original question:' +  r_d['question'] + ' ' + tokenizer.sep_token + 'Generated sparql query: ' for r_d in row_data
    ]
    label = [r_d['sparql_query'] + ' ' +tokenizer.eos_token for r_d in row_data]
    # tokenize the sentence
    input_tokenized = tokenizer(
        text=prompt_input,
        padding='max_length',
        truncation=True,
        max_length=max_length//2,
        add_special_tokens = False,
        return_tensors='pt'
    )
    label_tokenized = tokenizer(
        text=label,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        add_special_tokens = False,
        return_tensors='pt'
    )

    return {
        'sentence': input_tokenized['input_ids'].tolist(),
        'attention_mask': input_tokenized['attention_mask'].tolist(),
        'label': label_tokenized['input_ids'].tolist(),
        'label_mask': label_tokenized['attention_mask'].tolist()
    }
class NL2SQLDataset(Dataset):
    def __init__(self, 
                 tokenizer,
                 max_length=128,
                 status='train',
                 add_feature=False,
                 feature_type=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.status = status
        self.dataset = FLAGS.dataset
        os.makedirs('process_data',exist_ok=True)
        if add_feature:
            os.makedirs(f'process_data/{feature_type}',exist_ok=True)
            data_dir = f'process_data/{feature_type}'
        else:
            data_dir = 'process_data/orignal'

        
        data = load_data(data_dir,self.dataset,status)
        if add_feature:
            raise
        else:
            self.data = prepare_data_sentence_only(data,tokenizer,max_length)
        

    
    def __len__(self):
        return len(self.data['sentence'])
    
    def __getitem__(self, idx):
        return {
            'sentence': self.data['sentence'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'label': self.data['label'][idx],
            'label_mask': self.data['label_mask'][idx]
        }
@dataclass
class GenerationCollator:
    def __call__(self, batch):
        sentence = torch.tensor([item['sentence'] for item in batch]).long()
        attention_mask = torch.tensor([item['attention_mask'] for item in batch]).long()
        label = torch.tensor([item['label'] for item in batch]).long()
        label_mask = torch.tensor([item['label_mask'] for item in batch]).long()
        return {
            'input_ids':sentence,
            'attention_mask':attention_mask,
            'label':label,
            'label_mask':label_mask
        }