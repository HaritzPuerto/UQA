# Regularization of Distinct Strategies for Unsupervised Question Generation
We propose a novel regularization method for avoiding a bias toward a particular question generation strategy and modulating the process of generating individual words when a question is generated.

This repo provides codes for BERT-QA (by Hugginface), BERT-QG (Two assistants and Student), and Discriminator.

## Installation & Dependency
- We use transformers from HuggingFace for BERT.

      pip install transformers
      pip install torch==1.4
      pip install packaging
      pip install tensorboardX
      pip install boto3

## Pre-trained models
1. [Download][model_link] the trained models. This includes copy_type_assistant, lm_type_assistant, student, discriminator, and bert-base-uncased (for BERTQG).

[model_link]: https://drive.google.com/file/d/1AUFzWVWjjHMhDDca8iF8d9oUwCYhYsq-/view?usp=sharing "Download models"

2. Once you have downloaded the models, you should move them as follows (We assume your current directory is in UQA):
        
        unzip all_models.zip
        mkdir models
        cp -r all_models/discriminator.bin models/
        cp -r all_models/* BERTQG/models/
        rm -rf all_models

## Data
We provide the [data][data_link] generated by our models, and used for training our models.
- Once you download the data, unzip data.zip and you will get a directory called `data`.

        unzip data.zip

Inside the automatically created `data` folder, you will find the following files:

[data_link]: https://drive.google.com/file/d/1At8p5xE7FFoC5hbqWoSydhyU3FGPwArv/view?usp=sharing "Download data"

  - `QA_train_10k.json`: a training set for QA by generated our student model (final model).
  - `student_gen_10k.json`: a dataset, consists of (context, answer), to be used for generation of questions by the student.
  - `student_train_10k.json`: a training set for the student by generated our teacher (soft target).
  - `teacher_gen_10k.json`: a dataset, consists of (context, answer), to be used for generation of questions by the teacher (i.e., two assistants + regularization module).
  - `lm_type_train_10k.json / copy_type_train_10k.json`: a training set for the assistants.
  - `Disc_lm-type_40k.json / Disc_copy-type_40k.json`: a dataset used for training and evaluation for the discriminator.
  
Make sure that these datasets are in `data/` to execute the following scripts.

## Usage of scripts
  - `BERTQA/run_squad.py`: a script for testing the generated questions on a QA model (BERT-QA from HuggingFace). Note that we set dropout to 0, and use sequential sampling for discriminative results. With this setting, you should get a F1 score of about 60 on SQuAD 1.1 as in the paper.
  
        wget -P data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
        python BERTQA/run_squad.py \
            --model_type bert \
            --model_name_or_path bert-large-uncased-whole-word-masking \
            --do_train \
            --do_eval \
            --do_lower_case \
            --train_file data/QA_train_10k.json \
            --predict_file data/dev-v1.1.json \
            --per_gpu_train_batch_size 8 \
            --learning_rate 3e-5 \
            --num_train_epochs 2 \
            --max_seq_length 384 \
            --doc_stride 128 \
            --output_dir BERTQA/models/ \
            --logging_steps 5000 \
            --save_steps 5000 \

  - `student_generation.py`: a script for generating a QA training set using trained student model (final model).
  
        python3 student_generation.py \
            --bert_model BERTQG/models/bert-base-uncased \
            --student BERTQG/models/student/pytorch_model.bin \
            --input_file data/student_gen_10k.json \
            --output_file data/QA_train_10k.json \

  - `student_training.py`: a script for training the student model with the training set (soft target) generated by teacher.
  
        python3 student_training.py \
            --bert_model BERTQG/models/bert-base-uncased \
            --do_train \
            --train_file data/student_train_10k.json \
            --output_dir BERTQG/models/student/ \
            --num_train_epochs 2 \
            --train_batch_size 6 \
            --max_seq_length 512 \
            --doc_stride 450 \
            --max_query_length 42 \

  - `teacher_generation.py`: a script for generating the training set (soft target) using the trained teacher model. You may directly use the data generated by the teacher for QA as a regularization without student (more details in the paper).
  
        python3 teacher_generation.py \
            --bert_model BERTQG/models/bert-base-uncased \
            --lm_qg BERTQG/models/lm_type_assistant/pytorch_model.bin \
            --uqa_qg BERTQG/models/copy_type_assistant/pytorch_model.bin \
            --regul_model models/discriminator.bin \
            --input_file data/teacher_gen_10k.json \
            --output_file data/student_train_10k.json \
        
  - `assistant_training.py`: a script for training assistants. If the dataset is copy-type, the trained QG would be the copy-type assisant. If, on the other hand, the dataset is is LM-type, then the trained QG would be the LM-type assistant.
  
        python3 assistant_training.py \
            --bert_model BERTQG/models/bert-base-uncased \
            --do_train \
            --train_file data/lm_type_train_10k.json \
            --output_dir models/lm_type_assistant/ \
            --num_train_epochs 2 \
            --train_batch_size 6 \
            --max_seq_length 512 \
            --doc_stride 450 \
            --max_query_length 42 \         
  
  - `Discriminator_training.py`: a script for training the discriminator.
  
        python3 Discriminator_training.py
