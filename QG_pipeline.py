from BERTQG.token_generation import load_model, generate_token
from Regularization_module import Regularizer_Discriminator
from itertools import groupby
import numpy as np



class QuestionGeneration():
    def __init__(self, bert_model, lm_qg, uqa_qg, regularization=None):
        self.lm_qg, _, _ = load_model(bert_model, lm_qg)
        self.uqa_qg, self.tokenizer, self.device = load_model(bert_model, uqa_qg)
        #self.regularization = Freq_Regularization()
        self.regularization = Regularizer_Discriminator(regularization)
        
    def generate_question(self, context: str, ans: str, ans_start: str, list_question_tokens: list = []) -> str:
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
            list_token_classes = [-1] * len_initial_tokens  # 0 = UQA, 1 = LM
            list_qi_idx = [-1] * len_initial_tokens
            list_qi_probs = [-1] * len_initial_tokens

        # contains the classes of each token of the gen. question. Same len as list_question_tokens
        qi = qi_idx = qi_probs = None
        max_legnth = 50
        # generation finished when [SEP] is created
        while not self.__finished_generation(qi):
            question_text = " ".join(list_question_tokens)

            # Generate the toknes and probs of the ith query token using the lm and uqa models
            lm_qi_token, lm_qi_idx, lm_qi_probs = generate_token(self.lm_qg, self.tokenizer, self.device, context, question_text, ans, ans_start)
            uqa_qi_token, uqa_qi_idx, uqa_qi_probs = generate_token(self.uqa_qg, self.tokenizer, self.device, context, question_text, ans, ans_start)

            # Get token class to use
            qi_class = self.regularization.predict_class(context, ans, ans_start, question_text, list_question_tokens)
            # Get the predicted token
            if qi_class == 1:  # LM
                qi = lm_qi_token
                qi_idx = lm_qi_idx
                qi_probs = lm_qi_probs
            else:  # UQA
                qi = uqa_qi_token
                qi_idx = uqa_qi_idx
                qi_probs = uqa_qi_probs

            list_question_tokens.append(qi)
            list_token_classes.append(qi_class)
            list_qi_idx.append(qi_idx)
            list_qi_probs.append(qi_probs)

#             # for testing
#             print(list_question_tokens)
#             print(["UQA" if cls == 0 else "LM" for cls in list_token_classes])
#             print("\n")
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