# Guided Regularization between Copy and Paraphrase for Unsupervised Question Generation

### Download QG modules
LM: http://143.248.48.90:8080/shared/filebrowser/?token=e55f64532f5bafb00d25ac1f922b8215229f3971

UQA: http://143.248.48.90:8080/shared/filebrowser/?token=848397cc85a87771228cb2435958e32837032ace

Once you download two models, move them to correct directories as follows.

    mkdir BERTQG/models/lm_10k_QG
    mkdir BERTQG/models/uqa_10k_QG
    mv pytorch_model.bin BERTQG/models/lm_10k_QG/
    mv pytorch_model.bin BERTQG/models/uqa_10k_QG/
