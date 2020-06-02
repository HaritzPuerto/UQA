#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import transformers
from transformers import BertPreTrainedModel, BertModel
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import json
import numpy as np
import random


# In[ ]:


def data_gen_fun_from_jsonobj(jsonobj, tokenizer, flag,max_seq_length,doc_stride,max_query_length):
  jsonobj = data_partializer(jsonobj,tokenizer) # partialization
  processor = SquadV1Processor()
  examples = processor._create_examples(jsonobj['data'], set_type='train')
  features, dataset = squad_convert_examples_to_features(
      examples=examples,
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=True,
      return_dataset="pt",
      threads=1,
  )
  if flag == True: #UQA
    label = [0] * len(dataset)
  elif flag == False: #LM
    label = [1] * len(dataset)
  label_tensor = torch.tensor([f for f in label], dtype=torch.long)
  return dataset[:][:3], label_tensor

def data_gen_fun_from_path(filepath, filename,tokenizer, flag):
  processor = SquadV1Processor()
  examples = processor.get_train_examples(filepath, filename=filename)
  features, dataset = squad_convert_examples_to_features(
      examples=examples,
      tokenizer=tokenizer,
      max_seq_length=max_seq_length,
      doc_stride=doc_stride,
      max_query_length=max_query_length,
      is_training=True,
      return_dataset="pt",
      threads=1,
  )
  if flag == True: #UQA
    label = [0] * len(dataset)
  elif flag == False: #LM
    label = [1] * len(dataset)
  label_tensor = torch.tensor([f for f in label], dtype=torch.long)
  return dataset[:][:3], label_tensor

def data_partializer(jsonobj,tokenizer):
  for para in jsonobj['data']:
    for qas in para['paragraphs'][0]['qas']:
      tokenized_list = tokenizer.tokenize(qas['question'])
      if len(tokenized_list) <= 3:
        continue
      rnd_partial_index = random.randrange(2,len(tokenized_list)) #first one is wh-word, we need at least 2
      #new_partalized_text = ' '.join(tokenized_list[:rnd_partial_index])
      #new_partalized_text = new_partalized_text.replace(" ##","")
      new_partalized_text = tokenizer.convert_tokens_to_string(tokenized_list[:rnd_partial_index]) #use huggingface function instead
      qas['question'] = new_partalized_text
  return jsonobj

def data_perturbatation(tensor_data, pertub_prob):
  for tensor_ins in tensor_data[0]:
    for index in range(2,(tensor_ins == 102).nonzero()[0][0]):
      if random.random() < pertub_prob:
        tensor_ins[index] = 100
  return tensor_data


# In[ ]:


class BertDiscrim(BertPreTrainedModel):

    def __init__(self, config):
        super(BertDiscrim, self).__init__(config)
        self.bert = BertModel(config)
        #self.cls = BertPreTrainingHeads(config)
        self.config = config
        self.config_hiddensize = self.config.hidden_size
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.Linear_CLS_binary = nn.Linear(self.config.hidden_size, 2)
        self.softmax_fun = nn.Softmax(dim=1)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, label = None):
        sequence_output, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        pred = self.Linear_CLS_binary(pooled)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pred,label)
        return loss
    
    def forward_pred(self, input_ids, token_type_ids=None, attention_mask=None, label = None):
        sequence_output, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        pred = self.Linear_CLS_binary(pooled)
        values, indices = torch.max(pred, 1)
        return ((indices == label).sum().data.cpu().numpy() / label.data.cpu().numpy().shape[0])

    def forward_prob(self, input_ids, token_type_ids=None, attention_mask=None, label = None):
        sequence_output, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        pred = self.Linear_CLS_binary(pooled).data.cpu()
        return self.softmax_fun(pred).numpy().tolist()

class Regularizer_Discriminator():
  def __init__(self, modelpath):
    self.model_name_or_path = "bert-base-uncased"
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.max_seq_length = 384
    self.doc_stride = 128
    self.max_query_length = 64
    self.tokenizer = AutoTokenizer.from_pretrained(
        self.model_name_or_path,
        do_lower_case=True,
        cache_dir= None,
    )

    self.config = AutoConfig.from_pretrained(
        self.model_name_or_path,
        cache_dir= None,
    )

    self.output_model_file = modelpath
    self.model = BertDiscrim(config=self.config)
    self.model.load_state_dict(torch.load(self.output_model_file, map_location='cpu'))
    self.model.to(self.device)
  
  def predict_prob(self, jsonobj):
    dataset_pred, _ = data_gen_fun_from_jsonobj(jsonobj,self.tokenizer,True,self.max_seq_length,self.doc_stride,self.max_query_length)
    pred_id = dataset_pred[0]
    pred_mask = dataset_pred[1]
    pred_seg = dataset_pred[2]
    pred_data = TensorDataset(pred_id,pred_mask,pred_seg)
    pred_sampler = SequentialSampler(pred_data)
    pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=10)
    pred_epoch_iterator = tqdm(pred_dataloader, desc="dev_Iteration")
    result_list = []
    for batch in pred_epoch_iterator:
      batch = tuple(t.to(self.device) for t in batch)
      result = self.model.forward_prob(batch[0],batch[1], batch[2])
      result_list += result
    return np.asarray(result_list)


# In[ ]:

def main():

    ####Training


    # In[ ]:


    model_type = "bert"
    model_name_or_path = "bert-base-uncased"
    learning_rate = 3e-5
    num_train_epochs = 5
    max_seq_length = 384
    doc_stride = 128
    max_query_length = 64
    gradient_accumulation_steps = 1
    adam_epsilon = 1e-8
    warmup_steps = 0
    weight_decay = 0.0
    max_grad_norm = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        do_lower_case=True,
        cache_dir= None,
    )

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir= None,
    )


    # In[ ]:


    with open("./data/Disc_copy-type_40k.json","r", encoding="utf-8") as f:
      json_data_u = json.load(f)
    with open("./data/Disc_lm-type_40k.json","r", encoding="utf-8") as f:
      json_data_l = json.load(f)

    dataset_u, label_tensor_u = data_gen_fun_from_jsonobj(json_data_u,tokenizer,True,max_seq_length,doc_stride,max_query_length)
    dataset_l, label_tensor_l = data_gen_fun_from_jsonobj(json_data_l,tokenizer,False,max_seq_length,doc_stride,max_query_length)

    dataset_u = data_perturbatation(dataset_u,0.05) # adding pertubation
    dataset_l = data_perturbatation(dataset_l,0.05) # adding pertubation

    #train :35000 instances
    train_id = torch.cat((dataset_u[0][:35000],dataset_l[0][:35000]))
    train_mask = torch.cat((dataset_u[1][:35000],dataset_l[1][:35000]))
    train_seg = torch.cat((dataset_u[2][:35000],dataset_l[2][:35000]))
    train_label = torch.cat((label_tensor_u[:35000],label_tensor_l[:35000]))

    #eval 35000:
    dev_id = torch.cat((dataset_u[0][35000:],dataset_l[0][35000:]))
    dev_mask = torch.cat((dataset_u[1][35000:],dataset_l[1][35000:]))
    dev_seg = torch.cat((dataset_u[2][35000:],dataset_l[2][35000:]))
    dev_label = torch.cat((label_tensor_u[35000:],label_tensor_l[35000:]))

    train_data = TensorDataset(train_id,train_mask,train_seg, train_label)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=10)

    dev_data = TensorDataset(dev_id,dev_mask,dev_seg, dev_label)
    dev_sampler = SequentialSampler(dev_data)
    dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=10)


    # In[ ]:


    model = BertDiscrim.from_pretrained(model_name_or_path, from_tf=False, config=config, cache_dir= None)
    model.to(device)
    t_total = len(train_dataloader) // gradient_accumulation_steps * num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )


    # In[ ]:


    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    tr_loss, logging_loss = 0.0, 0.0 #logging is not used
    model.zero_grad()
    acc_max = -1

    for _ in range(num_train_epochs):
      epoch_iterator = tqdm(train_dataloader, desc="Iteration")
      for step, batch in enumerate(epoch_iterator):
        model.train()
        batch = tuple(t.to(device) for t in batch)
        loss = model(batch[0],batch[1], batch[2], batch[3])
        loss.backward()
        tr_loss += loss.item()
        if (step + 1) % gradient_accumulation_steps == 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
          optimizer.step()
          scheduler.step()
          model.zero_grad()
          global_step += 1

      #eval: no backward
      dev_epoch_iterator = tqdm(dev_dataloader, desc="dev_Iteration")
      acc_sum = []
      for batch in dev_epoch_iterator:
        batch = tuple(t.to(device) for t in batch)
        acc_sum.append(model.forward_pred(batch[0],batch[1], batch[2], batch[3]))
      acc_mean = sum(acc_sum) / len(acc_sum)
      print(acc_mean)
      if acc_max < acc_mean:
        acc_max = acc_mean
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = "./models/discriminator.bin"
        torch.save(model_to_save.state_dict(), output_model_file)

if __name__ == "__main__":
    main()