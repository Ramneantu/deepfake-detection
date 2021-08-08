#!/bin/sh
#python explainXception.py -mp "/home/deepfake/emre/repo/proj-4/Models/xception-models/xception_X-ray_e28_finetuning" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/Xray";
#python explainXception.py -mp "/home/deepfake/emre/repo/proj-4/Models/xception-models/xception_FF-compressed_e7_finetuning" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/c23";
#python explainXception.py -mp "/home/deepfake/emre/repo/proj-4/Models/xception-models/xception_FF-raw_e6_finetuning" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/c0";
#python explainXception.py -mp "/home/deepfake/emre/repo/proj-4/Models/xception-models/xception_HQ_e6_finetuning" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/HQ";

python explainFreq.py -mp "/home/deepfake/emre/repo/proj-4/Models/pretrained-freq-models/pretrained_NN_xray.pkl" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/smallData/Xray_small";
#python explainFreq.py -mp "/home/deepfake/emre/repo/proj-4/Models/pretrained-freq-models/pretrained_NN_c23.pkl" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/smallData/c23_small";
python explainFreq.py -mp "/home/deepfake/emre/repo/proj-4/Models/pretrained-freq-models/pretrained_NN_c0.pkl" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/small_data/c0_small";
#python explainFreq.py -mp "/home/deepfake/emre/repo/proj-4/Models/pretrained-freq-models/pretrained_NN_hq.pkl" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/smallData/HQ_small";

#python explainFreq.py -mp "/home/deepfake/emre/repo/proj-4/Models/pretrained-freq-models/pretrained_SVM_r_c23.pkl" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/c23";
#python explainFreq.py -mp "/home/deepfake/emre/repo/proj-4/Models/pretrained-freq-models/pretrained_SVM_r_c0.pkl" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/c0";
#python explainFreq.py -mp "/home/deepfake/emre/repo/proj-4/Models/pretrained-freq-models/pretrained_SVM_r_hq.pkl" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/HQ";
#python explainFreq.py -mp "/home/deepfake/emre/repo/proj-4/Models/pretrained-freq-models/pretrained_SVM_r_xray.pkl" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/Xray";

#/home/deepfake/emre/repo/proj-4/ExplainData
