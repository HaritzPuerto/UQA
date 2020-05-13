import torch
import torch.nn.functional as F
from .tokenization import whitespace_tokenize, BasicTokenizer, BertTokenizer
from .modeling import BertForGenerativeSeq
import collections
import logging
import os
import argparse

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class Data(object):
    """A single training/test example for the Squad dataset."""

    def __init__(self,
                 doc_tokens,
                 answers_text,
                 answer_start):
        self.doc_tokens = doc_tokens
        self.answers_text = answers_text
        self.answer_start = answer_start

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_pos):
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_pos = label_pos

def read_data(context, answer, answer_start):

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    datas = []
    answer_text = answer
    answer_start = answer_start

    answer_len = len(answer_text) 


    doc_tokens = []
    prev_is_whitespace = True
    for i, c in enumerate(context):
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if len(answer_text) == 1 and i == answer_start:
                doc_tokens.append("[HL]")
                doc_tokens.append(c)
                doc_tokens.append("[HL]")
                prev_is_whitespace = True
                continue
            elif answer_text[0] == c and i == answer_start:
                doc_tokens.append("[HL]")
                doc_tokens.append(c)
                prev_is_whitespace = False
                continue
            elif answer_text[-1] == c and i == answer_start + answer_len - 1:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                                
                doc_tokens.append("[HL]")
                prev_is_whitespace = False
                continue

            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False

    
    data = Data(
        doc_tokens=doc_tokens,
        answers_text=answer_text,
        answer_start=answer_start)
    datas.append(data)
        
    return datas


def convert_data_to_features(data, question_text, tokenizer, max_seq_length,
                                 doc_stride, max_query_length):
    
    features = []

    query_tokens = tokenizer.tokenize(question_text)

    all_doc_tokens = []                 
    tok_to_orig_index = []              
    orig_to_tok_index = []              
    
    for (i, token) in enumerate(data[0].doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)


    # The -5 accounts for [CLS], [HL], [HL], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - max_query_length - 5 

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)
    
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []


        tokens.append("[CLS]")
        segment_ids.append(0)
        
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

            is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                   split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(0)
        
        tokens.append("[SEP]")
        segment_ids.append(0)

        ## add query tokens right after context + [SEP]
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)

        label_pos = len(tokens)         

        tokens.append("[MASK]")
        segment_ids.append(2)            


        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # check [HL]
        check_symbol = 0
        for token_index, token_id in enumerate(input_ids):
            if token_id == 99:
                check_symbol += 1
                segment_ids[token_index] = 1
                continue

            elif check_symbol == 1:
                segment_ids[token_index] = 1
        
        if len(doc_spans) > 1 and check_symbol != 2:
            print("symbol error")
            if doc_span_index == len(doc_spans) - 1:
                if check_symbol == 0:
                    print(data[0].doc_tokens)
                    print(data[0].answers_text)
                    print(check_symbol)
                    print(input_ids)
                    print('HL error')
                    exit()
                    
                else:     
                    insert_num = max_tokens_for_doc - len(input_ids) + 3 #[CLS] [SEP] [MASK]

                    for num, pre_input_id in enumerate(pre_input_ids[-insert_num - 2:-2]):
                        input_ids.insert(num + 1,pre_input_id)

                    segment_ids = []
                    segment_ids = [0] * len(input_ids)

                    check_symbol = 0
                    for token_index, token_id in enumerate(input_ids):
                        if token_id == 99:
                            check_symbol += 1
                            segment_ids[token_index] = 1
                            continue

                        elif check_symbol == 1:
                            segment_ids[token_index] = 1

                    segment_ids[-1] = 2
            else:
                pre_input_ids = deepcopy(input_ids)
                continue

        input_mask = [1] * len(input_ids)
        
        
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_pos=label_pos
                ))
        break

    return features

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def load_model(bert_model, trained_model):
    """
    return model, tokenizer, device
    """
    modelpath = bert_model
    training_modelpath = trained_model

    device = torch.device("cuda")

    tokenizer = BertTokenizer.from_pretrained(modelpath)

    model_state_dict = torch.load(training_modelpath)
    print('Load', trained_model)
    model = BertForGenerativeSeq.from_pretrained(modelpath, state_dict=model_state_dict)
    model.eval()
    model.to(device)

    return model, tokenizer, device

def generate_token(model, tokenizer, device, wh_word, list_qi_idx, context, question_text, answer, answer_start):
    # penalize repetition token inside QG (need to input question history: list_qi_idx)

    data = read_data(context=context, answer=answer, answer_start=answer_start)

    features = convert_data_to_features(  
               data=data,
               question_text=question_text,
               tokenizer=tokenizer,
               max_seq_length=512,
               doc_stride=450,
               max_query_length=42)
    
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(device)
    
    label_pos = features[0].label_pos

    T = 1.0 # temperature
    P = 1.5 # penalty for repretition token

    with torch.no_grad():
        predictions = model(input_ids[0].unsqueeze(0), segment_ids[0].unsqueeze(0), input_mask[0].unsqueeze(0))

        ### Heuristics ###
        predictions[0][label_pos] /= T # temperature

        ans_tokens = answer.lower().split()
        if len(ans_tokens) == 1 and ans_tokens[0] in tokenizer.vocab:    
            ans_ids = tokenizer.convert_tokens_to_ids(ans_tokens)
            predictions[0][label_pos][ans_ids] = -float('inf') # set to zero for answer where the length of token is 1

        # if len(list_qi_idx) < 10:
            # banned_ids = tokenizer.convert_tokens_to_ids(['.', ',', "'", '?', '[SEP]'])
            # predictions[0][label_pos][banned_ids] = -float('inf') # set to zero for all banned_words
            
        banned_ids = tokenizer.convert_tokens_to_ids(['.', ',', "'", '?', '[SEP]'])
        predictions[0][label_pos][banned_ids] /= (15 / (len(list_qi_idx) + 1))

        if len(list_qi_idx) != 0:
            predictions[0][label_pos][list_qi_idx] /= P # penalty for repetition token
            # predictions[0][label_pos][list_qi_idx] = -float('inf')

        list_wh = ['what', 'which', 'where', 'when', 'who', 'why', 'how', 'whom', 'whose']
        if len(list_qi_idx) == 0 and wh_word.lower() in list_wh:
            list_wh.remove(wh_word.lower())
        list_wh_ids = tokenizer.convert_tokens_to_ids(list_wh)
        predictions[0][label_pos][list_wh_ids] = -float('inf') # set to zero for all wh-words except for input wh_word
        ### Heuristics ###

        probabilities = F.softmax(predictions[0][label_pos], 0).detach().cpu() # vocab size (30522)
            
        predicted_index = torch.argmax(probabilities).item()
        # score = probabilities[predicted_index].item()
        
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        predicted_text = predicted_token[0]

        return predicted_text, predicted_index, probabilities.tolist()
