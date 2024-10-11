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

    backbone = config.model_dict[FLAGS.model_name].from_pretrained(FLAGS.model_name)
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

    # load data
    train_dataset = config.dataset_dict[FLAGS.dataset](tokenizer,
                                                max_length=FLAGS.max_length,
                                                status='train',
                                                add_feature=FLAGS.add_feature,
                                                feature_type=FLAGS.feature_type,
                                                s_expression=FLAGS.use_s_expression)
    test_dataset = config.dataset_dict[FLAGS.dataset](tokenizer,
                                               max_length=FLAGS.max_length,
                                               status='test',
                                               add_feature=FLAGS.add_feature,
                                                feature_type=FLAGS.feature_type,
                                                s_expression=FLAGS.use_s_expression)
    train_dataloader = torch.utils.data.DataLoader(
                                                train_dataset,
                                                batch_size=FLAGS.train_batch_size,
                                                collate_fn=config.data_collator_dict[FLAGS.dataset](),
                                                shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
                                                test_dataset,
                                                batch_size=FLAGS.test_batch_size,
                                                collate_fn=config.data_collator_dict[FLAGS.dataset](),
                                                shuffle=False
    )
    if FLAGS.debug:
        train_dataset = test_dataset
        train_dataloader = test_dataloader
        logger.info("We are in debug mode, train_dataloader is the same as test_dataloader")
    logger.info('[data loaded]: train data size: {}, test data size: {}'.format(len(train_dataset),len(test_dataset)))

    # define optimizer
    # !omit: optimizer还传了很多其他的参数，可以check一下
    if FLAGS.fix_backbone:
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info('[Backbone fixed]')
    trainable_params_model = filter(lambda p: p.requires_grad, model.parameters()) if model is not None else []
    trainable_params_backbone = filter(lambda p: p.requires_grad, backbone.parameters())
    trainable_params = list(trainable_params_model) + list(trainable_params_backbone)
    logger.info(f'[Trainable parameters]: {len(trainable_params)}')

    if FLAGS.optimizer == 'sgd':
        optimizer = torch.optim.SGD(trainable_params,
                                    lr=FLAGS.lr,
                                    momentum=FLAGS.momentum,
                                    weight_decay=FLAGS.weight_decay)
    elif FLAGS.optimizer == 'adam':
        optimizer = torch.optim.Adam(trainable_params,
                                     lr=FLAGS.lr,
                                     weight_decay=FLAGS.weight_decay)
    elif FLAGS.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(trainable_params,
                                     lr=FLAGS.lr,
                                     weight_decay=FLAGS.weight_decay)
    else:
        raise ValueError('optimizer not supported')
    # define lr scheduler
    num_update_steps_per_epoch = len(train_dataloader) / FLAGS.gradient_accumulation_steps
    num_training_steps = FLAGS.max_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(FLAGS.warmup_ratio * num_training_steps)
    if FLAGS.lr_decay_type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                              T_max=num_training_steps,
                                                              eta_min=0)
    elif FLAGS.lr_decay_type == 'linear':
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                         num_training_steps,
                                                         eta_min=0)
    elif FLAGS.lr_decay_type == 'cosine_warmup':
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                         num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=num_training_steps)
    else:
        raise ValueError('lr decay type not supported')
    # Prepare everything with our `accelerator`.
    if model is not None:
        backbone,model, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
       backbone,model, optimizer, train_dataloader, test_dataloader
    )
    else:
        backbone, optimizer, train_dataloader, test_dataloader = accelerator.prepare(
       backbone, optimizer, train_dataloader, test_dataloader
    )
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / FLAGS.gradient_accumulation_steps)
    num_training_steps = FLAGS.max_epochs * num_update_steps_per_epoch
    if FLAGS.with_tracking:
        experiment_config = FLAGS.flag_values_dict()
        accelerator.init_trackers(f'{FLAGS.model_name+"_"+ FLAGS.dataset}', experiment_config, init_kwargs={"wandb": {"mode": "offline",'config':experiment_config}})
    # train
    best_em = 0
    patience_counter = 0
    total_batch_size = FLAGS.train_batch_size * accelerator.num_processes * FLAGS.gradient_accumulation_steps
    if accelerator.is_main_process:
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {FLAGS.max_epochs}")
        logger.info(f"  Instantaneous batch size per device = {FLAGS.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {FLAGS.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {num_training_steps}")
        logger.info(f"  reduction = {FLAGS.reduction}")
        if FLAGS.debug:
            logger.info(f"  !!!!!Debug mode!!!!!")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    
    for epoch in range(FLAGS.max_epochs):
        # model.train()
        if not FLAGS.fix_backbone:
            backbone.train()
        else:
            raise
        total_loss = []
        
        for step, batch in enumerate(train_dataloader):
            output = backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['label'],
            )
            loss = output.loss
            total_loss.append((float(loss.detach())))
            loss = loss / FLAGS.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % FLAGS.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
        logger.info(f'train/epoch: {epoch}, train/loss: {np.mean(total_loss)}')
        if FLAGS.with_tracking:
            accelerator.log(
                {
                    "train/epoch": epoch,
                    "train/loss": np.mean(total_loss),
                }
            )
        # eval
        # model.eval()
        if not FLAGS.fix_backbone:
            backbone.eval()
        total_loss = []
        all_preds = []
        all_labels = []
        evaluate.reset_metric()
        logger.info(f'***** evaluation *****')
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                outputs = backbone.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=FLAGS.max_length,
                    return_dict_in_generate=True,
                    output_logits=True,
                )
            logist = torch.stack(outputs.logits,dim=1)
            # output the logist shape and the label shape
            # logger.info(f'logist shape: {logist.size()}')
            # logger.info(f'label shape: {batch["label"].size()}')
            # loss = cross_entropy_loss(reduction=FLAGS.reduction)(logist.view(-1,logist.size(-1)),batch['label'].view(-1))
            # total_loss.append((float(loss.detach())))
            preds = outputs.sequences.cpu().numpy().tolist()
            # decode the output
            output_readable = tokenizer.batch_decode(preds,skip_special_tokens=True)
            # decode the label
            label_readable = tokenizer.batch_decode(batch['label'].cpu().numpy(),skip_special_tokens=True)
            # calculate the metrics
            evaluate.evaluate(output_readable,label_readable)
            all_preds.extend(output_readable)
            all_labels.extend(label_readable)
        if accelerator.is_main_process:
            report = evaluate.report()
            logger.info(f'test/epoch: {epoch}, "test/em": {report["exact_match"]}')
            logger.info(report)
        if FLAGS.with_tracking:
            accelerator.log(
                {
                    "test/epoch": epoch,
                    # "test/loss": np.mean(total_loss),
                    "test/em": report['exact_match'],
                }
            )
        # save the output
        with open(os.path.join(FLAGS.output_log_dir,f'{FLAGS.model_name.replace("/","-")}_{now_time}_output.json'),'a') as f:
            f.write(f'epoch: {epoch}\n')
            for output,label in zip(all_preds,all_labels):
                f.write(json.dumps({'output':output,'label':label})+'\n')
            f.write('\n\n')
        if report['exact_match'] > best_em:
            best_em = report['exact_match']
            patience_counter = 0
            if accelerator.is_main_process:
                logger.info(f'best em: {best_em}')
            # best_model_state_dict = model.state_dict()
            if not FLAGS.fix_backbone:
                best_backbone_state_dict = backbone.state_dict()

        else:
            patience_counter += 1
            if patience_counter >= FLAGS.patience:
                if accelerator.is_main_process:
                    logger.info(f'early stop at epoch: {epoch}')
                break
    accelerator.end_training()
    # model save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not FLAGS.debug  and FLAGS.save_model:
        os.makedirs(FLAGS.output_model_dir,exist_ok=True)
        save_path = os.path.join(FLAGS.output_model_dir,f'{FLAGS.model_name.replace("/","-")}_{now_time}.pt')
        # logger.info(f'[Model saved]')
        # logger.info(f'[Model saved path]: {save_path}')
        # accelerator.save(best_model_state_dict, save_path)
        if not FLAGS.fix_backbone:
            save_path = os.path.join(FLAGS.output_model_dir,f'{FLAGS.model_name.replace("/","-")}_backbone_{now_time}.pt')
            accelerator.save(best_backbone_state_dict, save_path)
            logger.info(f'[backboneModel saved]')
            logger.info(f'[backboneModel saved path]: {save_path}')
        logger.info(' [Best em]: {:.4f}'.format(best_em))

        
if __name__ == '__main__':
    app.run(main) 