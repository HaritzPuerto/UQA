# Guided Regularization between Copy and Paraphrase for Unsupervised Question Generation

### Download QG modules
LM: http://143.248.48.90:8080/shared/filebrowser/?token=a8725f86eea717c86a0e5d87972de7c3cbe3f248

UQA: http://143.248.48.90:8080/shared/filebrowser/?token=a72e9ccb7f6f1e24862b389001f8664dce656983

Once you download two models, move them to correct directories as follows.

    mkdir BERTQG/models/lm_10k_QG
    mkdir BERTQG/models/uqa_50k_QG
    mv pytorch_model.bin BERTQG/models/lm_10k_QG/
    mv pytorch_model_60000.bin BERTQG/models/uqa_50k_QG/
