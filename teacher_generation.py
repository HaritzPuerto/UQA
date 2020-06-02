import numpy as np
import json
import argparse
from tqdm import tqdm
from BERTQG.token_generation import load_model, generate_token
from Regularization_module import Regularizer_Discriminator
from itertools import groupby

class Freq_Regularization():
    def predict_class(self, list_token_classes):
        '''
        UQA class = 0
        LM class = 1
        '''
        if len(list_token_classes) == 0:
            return np.random.choice(2, 1, p=[0.5, 0.5])[0]

        prob_uqa = sum(list_token_classes)/len(list_token_classes)
        prob_lm = 1 - prob_uqa
        return np.random.choice(2, 1, p=[prob_uqa, prob_lm])[0]

class Random_Regularization():
    def predict_class(self):
        '''
        UQA class = 0
        LM class = 1
        '''
        return np.random.choice(2, 1, p=[0.5, 0.5])[0]

class QuestionGeneration():
    def __init__(self, bert_model, lm_qg, uqa_qg, regul=None):
        self.lm_qg, _, _ = load_model(bert_model, lm_qg)
        self.uqa_qg, self.tokenizer, self.device = load_model(bert_model, uqa_qg)
        self.regul = regul
        if self.regul == "rand":
            self.regularization = Random_Regularization()
        elif self.regul == "freq":
            self.regularization = Freq_Regularization()
        else: # disc path
            self.regularization = Regularizer_Discriminator(self.regul)
        
    def generate_question(self, context: str, ans: str, ans_start: str, wh_word: str, list_question_tokens: list = []) -> str:
        '''
        Input:
            - context
            - ans
            - ans_start
            - list_question_tokens: the history of generated question tokens. This is given for testing.
            The form of list_question_tokens is a list of (q_token, class), where class is 0 for uqa
            token and 1 for lm token
        Output:
            - The next generated question token
        '''
        # contains the tokens of the generated question
        if len(list_question_tokens) == 0:
            list_question_tokens = []
            list_token_classes = [] # 0 = UQA, 1 = LM
            list_qi_idx = [] 
            list_qi_probs = []
        else: # for testing
            len_initial_tokens = len(list_question_tokens)
            list_token_classes = [-1] * len_initial_tokens # 0 = UQA, 1 = LM
            list_qi_idx = [-1] * len_initial_tokens
            list_qi_probs = [-1] * len_initial_tokens

        # contains the classes of each token of the gen. question. Same len as list_question_tokens
        qi = qi_idx = qi_probs = None
        max_legnth = 50
        # generation finished when [SEP] is created
        while not self.__finished_generation(qi):
            question_text = " ".join(list_question_tokens).replace(' ##', '')
            
            # Get token class to use
            if self.regul == 'rand':
                # random
                qi_class = self.regularization.predict_class()
            elif self.regul == 'freq':
                # freq
                qi_class = self.regularization.predict_class(list_token_classes)
            else:
                # disc
                qi_class= self.regularization.predict_class(context,ans,ans_start,question_text,list_question_tokens)
            
            # Get the predicted token
            # Generate the toknes and probs of the ith query token using the lm and uqa models
            # penalize repetition token inside QG (need to input question history: list_qi_idx)
            if qi_class == 1: # LM
                qi, qi_idx, qi_probs = generate_token(self.lm_qg, self.tokenizer, self.device, wh_word, list_qi_idx, context, question_text, ans, ans_start)
            else: # UQA
                qi, qi_idx, qi_probs = generate_token(self.uqa_qg, self.tokenizer, self.device, wh_word, list_qi_idx, context, question_text, ans, ans_start)
    
            list_question_tokens.append(qi)
            list_token_classes.append(qi_class)
            list_qi_idx.append(qi_idx)
            list_qi_probs.append(qi_probs)
            
            if (len(list_question_tokens) > max_legnth):
                break
        
        # indices to keep
        list_idx = self.__remove_consecutive_repeated_tokens(list_question_tokens)

        # without [SEP] -> [:-1]
        list_question_tokens = [list_question_tokens[idx] for idx in list_idx][:-1]
        list_token_classes = [list_token_classes[idx] for idx in list_idx][:-1]
        list_qi_idx = [list_qi_idx[idx] for idx in list_idx][:-1]
        list_qi_probs = [list_qi_probs[idx] for idx in list_idx][:-1]
        
        assert len(list_question_tokens) == len(list_token_classes) == len(list_qi_idx) == len(list_qi_probs)

        return " ".join(list_question_tokens), list_token_classes, list_qi_idx, list_qi_probs
            
    def __finished_generation(self, question_token):
        return question_token =='[SEP]'
      
    def __remove_consecutive_repeated_tokens(self, list_tokens):
        '''
        Removes consecutive tokens.
        Sometimes the generated question is "when when did...",
        so we need to remove one when
        Output:
            - list of index that is not consecutive repeated
            - list of index to keep
        '''
        indices = range(len(list_tokens))
        return [list(group)[0][1] for key, group in groupby(zip(list_tokens, indices), lambda x: x[0])]
#     assert __remove_consecutive_repeated_tokens([1,1,1,1,1,1,2,3,4,4,5,1,2]) == [0, 6, 7, 8, 10, 11, 12]


def get_top_k(probs, k):
    top_k_indices = [np.argsort(-np.array(probs[token_idx]))[:k].tolist() for token_idx in range(len(probs))]
    top_k_probs = [sorted(probs[token_idx], reverse=True)[:k] for token_idx in range(len(probs))]
    return top_k_indices, top_k_probs

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert_model path.")

    parser.add_argument("--lm_qg", default=None, type=str, required=True,
                        help="lm_qg path.")

    parser.add_argument("--uqa_qg", default=None, type=str, required=True,
                        help="uqa_qg path.")

    parser.add_argument("--regul_model", default=None, type=str, required=True,
                        help="regularization_model path. Input rand for random, and freq for frequency-based model")

    parser.add_argument("--input_file", default=None, type=str, required=True, help="path for input file consists of (context, answer) pairs.")

    parser.add_argument("--output_file", default=None, type=str, required=True, help="path for generated data (soft target based training set for the student).")

    parser.add_argument("--top_k", default=10, type=int, required=False, help="top_k for probabililty distribution.")

    args = parser.parse_args()

    print(args)

    bert_model = args.bert_model
    lm_qg = args.lm_qg
    uqa_qg = args.uqa_qg
    regul_model = args.regul_model
    k = args.top_k

    QG = QuestionGeneration(bert_model, lm_qg, uqa_qg, regul_model)

    with open(args.input_file) as f:
        input_file = json.load(f)

        output = []
        for article in tqdm(input_file[:]):
            for paragraph in article['paragraphs']:
                for question in paragraph['qas']:
                    qid = question['id']
                    context = paragraph['context']
                    ans = question['answers'][0]['text']
                    ans_start = question['answers'][0]['answer_start']
                    wh_word = question['question']
                    assert context[ans_start:ans_start+len(ans)] == ans

                    question, history, indices, probs = QG.generate_question(context, ans, ans_start, wh_word)
                    top_k_indices, top_k_probs = get_top_k(probs, k)
                    assert len(question.split()) == len(top_k_indices) == len(top_k_probs)
                    paragraph = {
                                'id': qid,
                                'context': context,
                                'question': question.replace(' ##', ''),
                                'top_k_indices': top_k_indices,
                                'top_k_probs': top_k_probs,
                                'answers': ans,
                                'answer_start': ans_start
                                }
                    output.append(paragraph)

        with open(args.output_file, 'w') as f:
            json.dump(output, f)
    
if __name__ == '__main__':
    main()
