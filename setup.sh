#!/bin/bash
echo "Downloading UQA 10K model..." 

wget -O pytorch_model.bin "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/2ee97f79-b017-4bed-a8fa-1894073aba32/uqa_10k_QG.bin\?X-Amz-Algorithm\=AWS4-HMAC-SHA256\&X-Amz-Credential\=AKIAT73L2G45O3KS52Y5%2F20200506%2Fus-west-2%2Fs3%2Faws4_request\&X-Amz-Date\=20200506T022553Z\&X-Amz-Expires\=86400\&X-Amz-Signature\=8f795cb034a4008b43f296be365d38bb902c98d8e0155c9d8fe9ba8ab566a7a5\&X-Amz-SignedHeaders\=host\&response-content-disposition\=filename%20%3D%22uqa_10k_QG.bin%22"

mkdir BERTQG/models/uqa_10k_QG
mv pytorch_model.bin BERTQG/models/uqa_10k_QG/


echo "/n#################################"
echo "Downloading LM 10K model..." 
wget -O pytorch_model.bin "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/4c8fd5f3-0efd-41b0-ad87-d591d40fc9e0/lm_10k_QG.bin\?X-Amz-Algorithm\=AWS4-HMAC-SHA256\&X-Amz-Credential\=AKIAT73L2G45O3KS52Y5%2F20200506%2Fus-west-2%2Fs3%2Faws4_request\&X-Amz-Date\=20200506T022149Z\&X-Amz-Expires\=86400\&X-Amz-Signature\=bd26c7018321d8c1ba1829b188c38980c2cfacdfd9896edcdce78da0f43b396e\&X-Amz-SignedHeaders\=host\&response-content-disposition\=filename%20%3D%22lm_10k_QG.bin%22"

mkdir BERTQG/models/lm_10k_QG
mv pytorch_model.bin BERTQG/models/lm_10k_QG/



echo "/n#################################"
echo "Downloading discri_model_partial_perturb..." 
wget -O discri_model_partial_perturb.bin "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/eeaf3990-7df8-4f08-aaea-01b6994fe745/discri_model_partial_perturb.bin?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAT73L2G45O3KS52Y5%2F20200512%2Fus-west-2%2Fs3%2Faws4_request&X-Amz-Date=20200512T151610Z&X-Amz-Expires=86400&X-Amz-Signature=72a7e96012d42e5bad765b4c086e1084ca2167459200d859b3683c16ec1e06df&X-Amz-SignedHeaders=host&response-content-disposition=filename%20%3D%22discri_model_partial_perturb.bin%22"

mv discri_model_partial_perturb.bin models/
