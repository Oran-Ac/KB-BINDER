
'''dataset '''
from absl import flags
from torch.utils.data import Dataset
import torch
import os
import numpy as np
from dataclasses import dataclass
import json
FLAGS = flags.FLAGS
from accelerate.logging import get_logger
import re
logger = get_logger('my_logger')
from tqdm import tqdm
import csv
# load Freebase entity name
name_dir = os.path.join("/home/ctv651/sb/KB-BINDER/knowledge_source/Freebase/id2name_parts_disamb")
id2name_dict = {}
for file_name in tqdm(os.listdir(name_dir)):
    with open(os.path.join(name_dir, file_name), 'r') as rf:
        data_input = csv.reader(rf, delimiter="\t")
        for row in data_input:
            id2name_dict[row[0]] = row[2]
print("number of entities with names: ", len(id2name_dict))

def process_feature(item,feature_type,dataset):
    if feature_type == 'replaced':
        if dataset == 'webqsp_0107':
            # find the entity ids in the question(m.xxxxxxx)
            try:
                entity_ids = re.findall(r'm\.\w+',item['s_expression'])
            except:
                logger.info(item)
                raise
            # replace the entity ids with entity names
            for entity_id in entity_ids:
                if entity_id in id2name_dict:
                    item['s_expression'] = item['s_expression'].replace(entity_id,id2name_dict[entity_id])
            # find the entity ids in the sparql query(m.xxxxxxx)
            entity_ids = re.findall(r'm\.\w+',item['sparql_query'])
            # replace the entity ids with entity names
            for entity_id in entity_ids:
                if entity_id in id2name_dict:
                    item['sparql_query'] = item['sparql_query'].replace(entity_id,id2name_dict[entity_id])
            return {
                'question':item['question'],
                'sparql_query':item['sparql_query'],
                's_expression':item['s_expression']
            }

            
        else:
            raise
    else:
        raise

def prepare_data(data_dir,dataset,status,add_feature,feature_type):
    data = []
    raw_data_path = os.path.join(FLAGS.data_root_path,dataset)
    if dataset == 'webqsp_0107':
        raw_data_path = raw_data_path +f'.{status}.json'
        raw_data = json.load(open(raw_data_path))
        for item in raw_data:
            if item['question'] is None or item['sparql_query'] is None or item['s_expression'] is None:
                continue
            if add_feature:
                data.append(process_feature(item,feature_type,dataset))
            else:
                data.append({
                'question':item['question'],
                'sparql_query':item['sparql_query'],
                's_expression':item['s_expression']
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


def load_data(data_dir,dataset,status,add_feature,feature_type):
    data = []
    file_path = os.path.join(data_dir,dataset,status+'.jsonl')
    if not os.path.exists(file_path):
        return prepare_data(data_dir,dataset,status,add_feature,feature_type)
    with open(file_path,'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def clean_tokens(text,dataset_name):
        if dataset_name == 'webqsp_0107':
            drop_text = [
                "FILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))",
            ]
            for d_t in drop_text:
                text = text.replace(d_t,'')
        # remove '\n' and '\t'
        text = text.replace('\n', ' ').replace('\t', ' ')
        # remove multiple spaces
        text = re.sub(' +', ' ', text)
        # remove leading and trailing spaces
        text = text.strip()
        
        return text

def prepare_data_sentence_only(row_data,tokenizer,max_length,dataset_name):
    prompt_b = tokenizer.tokenize('Original question:')
    prompt_e = tokenizer.tokenize('Generated sparql query:')
    prompt_input_len = max_length//2 - len(prompt_b) - len(prompt_e) - 2
    label_len = max_length - 1
    logger.info(f'prompt_input_len: {prompt_input_len}, label_len: {label_len}')
    # just tokenize the sentence don't convert to ids
    question = [tokenizer.tokenize(clean_tokens(r_d['question'],dataset_name)) for r_d in row_data]
    query = [tokenizer.tokenize(clean_tokens(r_d['sparql_query'],dataset_name)) for r_d in row_data]
    logger.info(f'max question length: {max([len(q) for q in question])}')
    logger.info(f'max query length: {max([len(q) for q in query])}')
    prompt_input = [
         tokenizer.bos_token +  ' Original question:' +  ' '.join(q[:prompt_input_len]) + ' ' + tokenizer.sep_token + 'Generated sparql query: ' for q in question
    ]
    label = [' '.join(q[:label_len]) + ' ' +tokenizer.eos_token for q in query]

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

def prepare_data_s_expression(row_data,tokenizer,max_length,dataset_name):
    prompt_b = tokenizer.tokenize('Original question:')
    prompt_e = tokenizer.tokenize('Generated formulate:')
    prompt_input_len = max_length//2 - len(prompt_b) - len(prompt_e) - 2
    label_len = max_length - 1
    logger.info(f'prompt_input_len: {prompt_input_len}, label_len: {label_len}')
    # just tokenize the sentence don't convert to ids
    question = [tokenizer.tokenize(clean_tokens(r_d['question'],dataset_name)) for r_d in row_data]
    query = [tokenizer.tokenize(clean_tokens(r_d['s_expression'],dataset_name)) for r_d in row_data]
    logger.info(f'max question length: {max([len(q) for q in question])}')
    logger.info(f'max query length: {max([len(q) for q in query])}')
    prompt_input = [
         tokenizer.bos_token +  ' Original question:' +  ' '.join(q[:prompt_input_len]) + ' ' + tokenizer.sep_token + 'Generated formulate: ' for q in question
    ]
    label = [' '.join(q[:label_len]) + ' ' +tokenizer.eos_token for q in query]
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
                 feature_type=None,
                 s_expression=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.status = status
        self.dataset = FLAGS.dataset
        os.makedirs('process_data',exist_ok=True)
        if add_feature:
            os.makedirs(f'process_data/{feature_type}',exist_ok=True)
            data_dir = f'process_data/{feature_type}'
        else:
            data_dir = 'process_data/original'

        
        data = load_data(data_dir,self.dataset,status,add_feature,feature_type)
        if s_expression:
            self.data = prepare_data_s_expression(data,tokenizer,max_length,self.dataset)
        else:
            self.data = prepare_data_sentence_only(data,tokenizer,max_length,self.dataset)
        

    
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