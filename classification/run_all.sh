#!/bin/bash

nohup python classify_xception.py -d X-ray -l -e 30 --early_stopping 10 --lr_decay 3 -bs 32

nohup python classify_xception.py -d HQ -l -e 30 --early_stopping 5 -bs 32

nohup python classify_xception.py -d FF-raw -l -e 30 --early_stopping 5 -bs 32

nohup python classify_xception.py -d FF-compressed -l -e 30 --early_stopping 5 -bs 32

