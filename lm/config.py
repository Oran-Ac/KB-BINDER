from transformers import T5ForConditionalGeneration 
from data import *
model_dict = {
    'Salesforce/codet5-base':T5ForConditionalGeneration,
    'Salesforce/codet5-large':T5ForConditionalGeneration,
}

dataset_dict = {
    'webqsp_0107': NL2SQLDataset,
}
data_collator_dict = {
    'webqsp_0107': GenerationCollator
}