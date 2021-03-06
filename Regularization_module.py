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
    #squad_convert_examples_to_features,
)
from squadprocessor import squad_convert_examples_to_features,SquadResult,SquadV1Processor,SquadV2Processor
#from transformers.data.processors.squad import (
    #SquadResult,
    #SquadV1Processor,
    #SquadV2Processor,
#)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import json
import numpy as np


# In[ ]:


def data_gen_fun_from_jsonobj(
    jsonobj, tokenizer, flag, max_seq_length, doc_stride, max_query_length
):
    processor = SquadV1Processor()
    examples = processor._create_examples(jsonobj["data"], set_type="train")
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
    if flag == True:  # UQA
        label = [0] * len(dataset)
    elif flag == False:  # LM
        label = [1] * len(dataset)
    label_tensor = torch.tensor([f for f in label], dtype=torch.long)
    return dataset[:][:3], label_tensor


def data_gen_fun_from_path(filepath, filename, tokenizer, flag):
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
    if flag == True:  # UQA
        label = [0] * len(dataset)
    elif flag == False:  # LM
        label = [1] * len(dataset)
    label_tensor = torch.tensor([f for f in label], dtype=torch.long)
    return dataset[:][:3], label_tensor


# In[ ]:


class BertDiscrim(BertPreTrainedModel):
    def __init__(self, config):
        super(BertDiscrim, self).__init__(config)
        self.bert = BertModel(config)
        # self.cls = BertPreTrainingHeads(config)
        self.config = config
        self.config_hiddensize = self.config.hidden_size
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.Linear_CLS_binary = nn.Linear(self.config.hidden_size, 2)
        self.softmax_fun = nn.Softmax(dim=1)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, label=None):
        sequence_output, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        pred = self.Linear_CLS_binary(pooled)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pred, label)
        return loss

    def forward_pred(
        self, input_ids, token_type_ids=None, attention_mask=None, label=None
    ):
        sequence_output, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        pred = self.Linear_CLS_binary(pooled)
        values, indices = torch.max(pred, 1)
        return (
            indices == label
        ).sum().data.cpu().numpy() / label.data.cpu().numpy().shape[0]

    def forward_prob(
        self, input_ids, token_type_ids=None, attention_mask=None, label=None
    ):
        sequence_output, pooled = self.bert(input_ids, token_type_ids, attention_mask)
        pred = self.Linear_CLS_binary(pooled).data.cpu()
        return self.softmax_fun(pred).numpy().tolist()


class Regularizer_Discriminator:
    def __init__(self, modelpath):
        self.model_name_or_path = "bert-base-uncased"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_length = 384
        self.doc_stride = 128
        self.max_query_length = 64
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, do_lower_case=True, cache_dir=None,
        )

        self.config = AutoConfig.from_pretrained(
            self.model_name_or_path, cache_dir=None,
        )

        self.output_model_file = modelpath
        self.model = BertDiscrim(config=self.config)
        self.model.load_state_dict(
            torch.load(self.output_model_file, map_location="cpu")
        )
        self.model.to(self.device)
        self.model.eval()
    def predict_class(self, context, ans, ans_start, question_text, list_token_classes):
        if len(list_token_classes) == 0:
            return np.random.choice(2, 1, p=[0.5, 0.5])[0]
        jsonobj = self.__convert2squad(context, ans, ans_start, question_text)
        dataset_pred, _ = data_gen_fun_from_jsonobj(
            jsonobj,
            self.tokenizer,
            True,
            self.max_seq_length,
            self.doc_stride,
            self.max_query_length,
        )
        with torch.no_grad():
            pred_id = dataset_pred[0]
            pred_mask = dataset_pred[1]
            pred_seg = dataset_pred[2]
            pred_data = TensorDataset(pred_id, pred_mask, pred_seg)
            pred_sampler = SequentialSampler(pred_data)
            pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=10)
            #pred_epoch_iterator = tqdm(pred_dataloader, desc="dev_Iteration")
            result_list = []
            for batch in pred_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                result = self.model.forward_prob(batch[0], batch[1], batch[2])
                result_list += result
            qi_probs = np.asarray(result_list)
            # qi_probs: [batchsize,2] ndarray
            qi_class = np.argmax(
                qi_probs, axis=1
            )  # qi_class: [batchsize,] ndarray 0:should generate UQA, 1:should generate LM
            qi_class = qi_class[0]  # For now, let's assume batch = 1
            return 1 - qi_class #0: UQA, 1: LM -> 0: choose LM, 1: choose UQA
    
    def __convert2squad(self, context: str, answer: str, ans_start: int,  question: str) -> dict:
        '''
        Create a SQuAD instance
        Inputs:
            - context: paragrah
            - answer
            - ans_start
            - question: might not be the full question (we are generating questions token by token)
        Returns:
            - squad instance
        '''
        squad = {'data': [], 'version': '1.0'}
        squad['data'].append({'title': 'title', 'paragraphs': [{'context': context,
                                              'qas': [{'answers': [{'answer_start': ans_start, 'text': answer}],
                                                       'question': question,
                                                       'id': 0}]}]})
        return squad

