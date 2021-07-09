#!/bin/sh
python explainXception.py -mp "/home/deepfake/emre/repo/proj-4/Models/xception-models/xception_FF-compressed_e7_finetuning" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/c23";
python explainXception.py -mp "/home/deepfake/emre/repo/proj-4/Models/xception-models/xception_FF-raw_e6_finetuning" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/c0";
python explainXception.py -mp "/home/deepfake/emre/repo/proj-4/Models/xception-models/xception_HQ_e6_finetuning" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/HQ";
python explainXception.py -mp "/home/deepfake/emre/repo/proj-4/Models/xception-models/xception_X-ray_e28_finetuning" -ed "/home/deepfake/emre/repo/proj-4/ExplainData/Xray";




/home/deepfake/emre/repo/proj-4/ExplainData