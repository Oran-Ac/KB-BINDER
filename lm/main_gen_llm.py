from absl import app
from absl import flags
from tqdm.auto import tqdm
from transformers import AutoTokenizer,get_cosine_schedule_with_warmup
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed,DistributedDataParallelKwargs
import json
import random
import numpy as np
import torch
import torch.nn as nn
import logging
import datetime
import math
import os
from evaluate import SparqlEvaluator
# import self-defined modules
import config
from loss import *
import transformers

logger = get_logger('my_logger')

FLAGS = flags.FLAGS
# parameters for data
flags.DEFINE_string('dataset','sst2','dataset name')
flags.DEFINE_string('data_root_path','./data/','RAW dataset root path') 
flags.DEFINE_float('data_ratio',1.0,'data ratio')
# parameters for model
flags.DEFINE_string('model_name','bert-base-uncased','model name')
flags.DEFINE_float('dropout_rate',0.0,'dropout rate')
# parameters for training
flags.DEFINE_bool('fix_backbone',False,'fix backbone')
flags.DEFINE_integer('seed',0,'random seed')
flags.DEFINE_integer('train_batch_size',32,'batch size')
flags.DEFINE_integer('eval_batch_size',32,'batch size')
flags.DEFINE_integer('test_batch_size',32,'batch size')
flags.DEFINE_integer('max_length',128,'max length')
flags.DEFINE_integer('eval_steps',100,'eval steps')
flags.DEFINE_integer('max_epochs',10,'max epochs')
flags.DEFINE_integer('patience',3,'patience')
flags.DEFINE_integer('gradient_accumulation_steps',1,'gradient accumulation steps')
flags.DEFINE_bool('debug',False,'debug mode')
flags.DEFINE_string('reduction','mean','reduction')
# parameters for optimizer
flags.DEFINE_string('optimizer','sgd','optimizer')
flags.DEFINE_string('lr_decay_type','cosine','lr decay type')
flags.DEFINE_float('momentum',0.9,'momentum')
flags.DEFINE_float('lr',1e-5,'learning rate')
flags.DEFINE_float('weight_decay',1e-8,'weight decay')
flags.DEFINE_float('warmup_ratio',0.1,'warmup ratio')
# save
flags.DEFINE_string('output_model_dir','./saved_model','output directory')
flags.DEFINE_string('output_log_dir','./log','output directory')
flags.DEFINE_bool('save_model',True,'save model')
# accelerator
flags.DEFINE_bool('with_tracking',True,'with tracking')
# feature
flags.DEFINE_bool('add_feature',False,'only sentence2sentence')
flags.DEFINE_string('feature_type',None,'feature type') #e.g., [schema,replaced] #?it's shouldn't be String, it should be List  
# s expression
flags.DEFINE_bool('use_s_expression',False,'use s expression')
# few-shot
flags.DEFINE_integer('few_shot',0,'shot')
def set_seed(seed):
    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main(argv):
    # parse flags
    app.parse_flags_with_usage(argv)
    # set seed
    set_seed(FLAGS.seed)
    FLAGS.output_model_dir = os.path.join(FLAGS.output_model_dir,FLAGS.dataset)
    FLAGS.output_log_dir = os.path.join(FLAGS.output_log_dir,FLAGS.dataset)
    if not FLAGS.fix_backbone:
        FLAGS.output_log_dir = os.path.join(FLAGS.output_log_dir,'not_fix_backbone')
        FLAGS.output_model_dir = os.path.join(FLAGS.output_model_dir,'not_fix_backbone')

    # create output directory
    if not os.path.exists(FLAGS.output_model_dir):
        os.makedirs(FLAGS.output_model_dir,exist_ok=True)
    if not os.path.exists(FLAGS.output_log_dir):
        os.makedirs(FLAGS.output_log_dir,exist_ok=True)
    
    # define accelerator
    accelerator = Accelerator(log_with="wandb") if FLAGS.with_tracking else Accelerator()
    if accelerator.is_main_process:
        now_time = datetime.datetime.now().strftime('%m-%d_%H-%M%-S')
        # configure logging
        fileHeader = logging.FileHandler(
                filename = os.path.join(FLAGS.output_log_dir,f'{FLAGS.model_name.replace("/","-")}_{now_time}.log'), 
                mode = 'w')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        fileHeader.setFormatter(formatter)
        # call inside the logger:https://github.com/huggingface/accelerate/blob/658492fb410396d8d1c1241c1cc2412a908b431b/src/accelerate/logging.py#L112
        logger.logger.addHandler(fileHeader)
        logger.logger.setLevel(logging.DEBUG if FLAGS.debug else logging.INFO)


        # logging the flags
        logger.info(json.dumps(FLAGS.flag_values_dict(),indent=4))
        logger.info(accelerator.state, main_process_only=False) 

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name)
    evaluate = SparqlEvaluator()
    if FLAGS.add_feature:
        if FLAGS.feature_type == 'replaced':
            model = None
        else:
            raise
    else:
        model = None
    
    logger.info(f'[Model structure]:\n {model}')


    assert FLAGS.test_batch_size == 1
    test_dataset = config.dataset_dict_few_shot[FLAGS.dataset](tokenizer,
                                            max_length=FLAGS.max_length,
                                            status='test',
                                            add_feature=FLAGS.add_feature,
                                                feature_type=FLAGS.feature_type,
                                                s_expression=FLAGS.use_s_expression,
                                                few_shot=FLAGS.few_shot)
    test_dataloader = torch.utils.data.DataLoader(
                                                test_dataset,
                                                batch_size=FLAGS.test_batch_size,
                                                shuffle=False
    )




    pipeline = transformers.pipeline(
        "text-generation",
        model=FLAGS.model_name,
        torch_dtype=torch.float16,
        device_map="auto",

    )
   
    all_preds = []
    all_labels = []
    for batch in tqdm(test_dataloader,total=len(test_dataloader)):
        preds = pipeline(
            batch['sentence'],
            max_new_tokens=FLAGS.max_length,
            num_return_sequences=1,
            do_sample=True,
            #temperature=0.5,
            top_k=50,
            top_p=0.95,
            #num_beams=1,
            #no_repeat_ngram_size=2,
            early_stopping=True,
            return_full_text=False # only return the generated text
        )
        # find the first '\n'
        preds = preds[0][0]['generated_text'].split('\n')[0]
        evaluate.evaluate(preds,batch['label'])
        all_preds.extend(preds)
        all_labels.extend(batch['label'])
    if accelerator.is_main_process:
        report = evaluate.report()
        logger.info(f'"test/em": {report["exact_match"]}')
        logger.info(report)
    if FLAGS.with_tracking:
        accelerator.log(
            {
                "test/em": report['exact_match'],
            }
        )
    # save the output
    with open(os.path.join(FLAGS.output_log_dir,f'{FLAGS.model_name.replace("/","-")}_{now_time}_output.json'),'a') as f:
        for output,label in zip(all_preds,all_labels):
            f.write(json.dumps({'output':output,'label':label})+'\n')
        f.write('\n\n')
        
if __name__ == '__main__':
    app.run(main) 