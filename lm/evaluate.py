import re

import json
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu



class SparqlEvaluator:
    def __init__(self):

        self.reset_metric()
        
    def clean_tokens(self, text):
        # remove '\n' and '\t'
        text = text.replace('\n', ' ').replace('\t', ' ')
        # remove multiple spaces
        text = re.sub(' +', ' ', text)
        # remove leading and trailing spaces
        text = text.strip()
        return text

    def evaluate(self, preds, labels):
        preds = [self.clean_tokens(pred) for pred in preds]
        labels = [self.clean_tokens(label) for label in labels]
        
        self.collect_ngram(preds)
        self.compute_bleu(preds, labels)
        self.compute_exact_match(preds, labels)
        self.sent_cnt += len([pred for pred in preds if len(pred) > 0])

    def collect_ngram(self, strs):
        for str in strs:
            str = str.split()
            for k in range(1, 5):
                dist_k = f'dist@{k}'
                for token in ngrams(str, k):
                    self.metric[dist_k].add(token)

    def compute_bleu(self, preds, labels):
        for pred, label in zip(preds, labels):
            pred, label = pred.split(), [label.split()]
            for k in range(4):
                weights = [0] * 4
                weights[k] = 1
                self.metric[f'bleu@{k + 1}'] += sentence_bleu(label, pred, weights)
    
    def compute_exact_match(self, preds, labels):
        for pred, label in zip(preds, labels):
            if pred == label:
                self.metric['exact_match'] += 1


    def report(self):
        report = {}
        for k, v in self.metric.items():
            if self.sent_cnt == 0:
                report[k] = 0
            else:
                if 'dist' in k:
                    v = len(v)
                report[k] = v / self.sent_cnt
        report['sent_cnt'] = self.sent_cnt
        return report

    def reset_metric(self):
        self.metric = {
            'bleu@1': 0,
            'bleu@2': 0,
            'bleu@3': 0,
            'bleu@4': 0,
            'dist@1': set(),
            'dist@2': set(),
            'dist@3': set(),
            'dist@4': set(),
            'exact_match': 0
        }
        self.sent_cnt = 0