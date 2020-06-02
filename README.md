# Regularization of Distinct Strategies for Unsupervised Question Generation
This repo provides codes for BERT-QA (by Hugginface), BERT-QG (Two assistants and Student), and Discriminator.

## Installation & Dependency
- We use transformers by HuggingFace for BERT.

      pip install transformers
      pip install torch==1.4

## Pre-trained models
1. [Download][model_link] trained models via the following link. This includes copy_type_assistant, lm_type_assistant, student, discriminator, and bert-base-uncased (for BERTQG).

[model_link]: https://s3.us-west-2.amazonaws.com/secure.notion-static.com/b3dba4b2-4ff6-42f3-bc89-22f80b17a448/models.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200602%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200602T061832Z&X-Amz-Expires=86400&X-Amz-Signature=2940b25a3fafd7f35ed37abe97afa6870b647bb0f7b8e9c3aa420e8a06a1927c&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22all_models.zip%22 "Download models"

2. Once you download models, you should move them as follows (We assume your current directory is in UQA):
        
        unzip all_models.zip
        mkdir models
        mv all_models/discriminators.bin $CODE_DIR/models/
        mv all_models/* $CODE_DIR/BERTQG/models/
        rm -rf all_models

## Data
We provide [data][data_link] generated by our models, and used for training our models.
- Once you download data, unzip data.zip and get a directory `data`.

        unzip data.zip

The descriptions of each data are as follows:

[data_link]: https://s3.us-west-2.amazonaws.com/secure.notion-static.com/97c1941f-ff53-4316-9b5d-65a446c5447c/data.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200602%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200602T080239Z&X-Amz-Expires=86400&X-Amz-Signature=bbe347b1b3d266da7824598f193dde74edf1dfaa81d3bc0509cd7ee71c49488a&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22data.zip%22 "Download data"

  - `QA_train_10k.json`: a training set for QA by generated our student model (final model).
  - `student_gen_10k.json`: a dataset, consists of (context, answer), to be used for generation of questions by the student.
  - `student_train_10k.json`: a training set for the student by generated our teacher (soft target).
  - `teacher_gen_10k.json`: a dataset, consists of (context, answer), to be used for generation of questions by the teacher (i.e., two assistants + regularization module).
  - `lm_type_train_10k.json / copy_type_train_10k.json`: a training set for the assistants.
  - `Disc_lm-type_40k.json / Disc_copy-type_40k.json`: a dataset used for training and evaluation for the discriminator.
  
Make sure that these data are in `data/` to execute following scripts.

## Usage of scripts
  - `BERTQA/run_squad.py`: a script for testing the generated question on QA model (BERT by HuggingFace). Note that we use dropout as 0, and sequential sampling for discriminative results. With this setting, you should get a F1 score of about 60 on SQuAD 1.1 as in the paper.
  
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

  - `student_generation.py`: a script for generating QA training set using trained student model (final model).
  
        python3 student_generation.py \
            --bert_model BERTQG/models/bert-base-uncased \
            --student BERTQG/models/student/pytorch_model.bin \
            --input_file data/student_gen_10k.json \
            --output_file data/QA_train_10k.json \

  - `student_training.py`: a script for training student model with training set (soft target) generated by teacher.
  
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

  - `teacher_generation.py`: a script for generating training set (soft target) using trained teacher model. You may directly use the data generated by this for QA as a (regularization without student, see our paper).
  
        python3 teacher_generation.py \
            --bert_model BERTQG/models/bert-base-uncased \
            --lm_qg BERTQG/models/lm_type_assistant/pytorch_model.bin \
            --uqa_qg BERTQG/models/copy_type_assistant/pytorch_model.bin \
            --regul_model models/discriminator.bin \
            --input_file data/teacher_gen_10k.json \
            --output_file data/student_train_10k.json \
        
  - `assistant_training.py`: a script for training assistants. The type is decided by the data (either copy-type or LM-type).
  
        python3 assistant_training.py \
            --bert_model models/bert-base-uncased \
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
