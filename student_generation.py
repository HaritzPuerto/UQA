import numpy as np
import json
import argparse
from tqdm import tqdm
from BERTQG.token_generation import load_model, generate_token
from itertools import groupby

class QuestionGeneration():
    def __init__(self, bert_model, final_qg):
        self.final_qg, self.tokenizer, self.device = load_model(bert_model, final_qg)
        
    def generate_question(self, context: str, ans: str, ans_start: str, wh_word: str, list_question_tokens: list = []) -> str:
        '''
        Input:
            - context
            - ans
            - ans_start
            - list_question_tokens: the history of generated question tokens. This is given for testing.
        Output:
            - The next generated question token
        '''
        # contains the tokens of the generated question
        if len(list_question_tokens) == 0:
            list_question_tokens = []
            list_qi_idx = [] 
            list_qi_probs = []
        else: # for testing
            len_initial_tokens = len(list_question_tokens)
            list_qi_idx = [-1] * len_initial_tokens
            list_qi_probs = [-1] * len_initial_tokens

        # contains the classes of each token of the gen. question. Same len as list_question_tokens
        qi = qi_idx = qi_probs = None
        max_legnth = 50
        # generation finished when [SEP] is created
        while not self.__finished_generation(qi):
            question_text = " ".join(list_question_tokens).replace(' ##', '')
            
            qi, qi_idx, qi_probs = generate_token(self.final_qg, self.tokenizer, self.device, wh_word, list_qi_idx, context, question_text, ans, ans_start)
    
            list_question_tokens.append(qi)
            list_qi_idx.append(qi_idx)
            list_qi_probs.append(qi_probs)
            
            if (len(list_question_tokens) > max_legnth):
                break
        
        # indices to keep
        list_idx = self.__remove_consecutive_repeated_tokens(list_question_tokens)

        # without [SEP] -> [:-1]
        list_question_tokens = [list_question_tokens[idx] for idx in list_idx][:-1]
        list_qi_idx = [list_qi_idx[idx] for idx in list_idx][:-1]
        list_qi_probs = [list_qi_probs[idx] for idx in list_idx][:-1]
        
        assert len(list_question_tokens) == len(list_qi_idx) == len(list_qi_probs)

        return " ".join(list_question_tokens), list_qi_idx, list_qi_probs
            
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

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="bert_model path.")

    parser.add_argument("--student", default=None, type=str, required=True,
                        help="student path.")

    parser.add_argument("--input_file", default=None, type=str, required=True, help="path for input file consists of (context, answer) pairs.")

    parser.add_argument("--output_file", default=None, type=str, required=True, help="path for generated data for QA.")

    args = parser.parse_args()

    print(args)

    bert_model = args.bert_model
    student = args.student
    
    QG = QuestionGeneration(bert_model, student)

    with open(args.input_file) as f:
        input_file = json.load(f)

        output = {'data': [], 'version': 'v1.1'}
        for article in tqdm(input_file[:10]):
            for paragraph in article['paragraphs']:
                for question in paragraph['qas']:
                    qid = question['id']
                    context = paragraph['context']
                    ans = question['answers'][0]['text']
                    ans_start = question['answers'][0]['answer_start']
                    wh_word = question['question']
                    assert context[ans_start:ans_start+len(ans)] == ans 

                    question, indices, probs = QG.generate_question(context, ans, ans_start, wh_word)
        
                    # SQuAD Format
                    paragraph = {'title': 'title',
                                'paragraphs': [{'context': context, 
                                'qas': [{'question': question.replace(' ##', ''),
                                        'id': qid, 
                                        'answers': [{'text': ans, 'answer_start': ans_start}]
                                        }]}]}
                    output['data'].append(paragraph)

        with open(args.output_file, 'w') as f:
            json.dump(output, f)
    
if __name__ == '__main__':
    main()
