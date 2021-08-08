# CNN Classifier

The XceptionNet classifier from our project is based on the [FaceForensics++ implementation](https://github.com/ondyari/FaceForensics). We trained models on our four datasets: FF++ raw, FF++ compressed, HQ and X-Ray dataset. All the datasets and the pretrained models are available to download under [this link](https://syncandshare.lrz.de/getlink/fiWJXehJXqESXNxopXNUYyNA/).
To train your own models, you will need to create the following folder structure:

## Run the classifier

**Setup**:
- Install required modules via `requirement.txt` file
- Additionally, install pytorch by `pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 -f https://download.pytorch.org/whl/torch_stable.html`
- Use one of our datasets or create your own, which adheres to the folder structure: the dataset should have a 'train', 'test' and 'val' subfolder, and each of these should have a 'real' and 'fake' subfolder. Datasets should be stored in `proj-4/data`
- Add the ImageNet model as `proj-4/data/models/xception-b5690688.pth`
- Run the classifier on a dataset
```shell
python classify_xception.py
-d <dataset: 'FF-raw', 'FF-compressed', 'X-ray', 'HQ'
-mi (optional) <path to a model file
-l (optional) <set to train the whole net (finetuning), not only the last layer (feature extraction)
-e (optinal) <maximum number of epochs. May stop sooner because of early stopping
--lr_decay (optional) <number of epochs before reducing learning rate. Default: inf
--early_stopping (optional) <number of epochs without improvement before stopping
-bs (optional) <batch size
```
**Remarks**:
- The number of epochs for `lr_decay` and `early_stopping` refers to the epochs without any improvement w.r.t. validation loss.
- If `-mi` is specified, the network is in test mode and only computes the accuracy on test set. Otherwise, the network is being trained starting from an ImageNet model.
- If a GPU is available, it will be used by the CNN.

**Output**:
- A log of the epochs and the results is stored in `data/experiments.log`
- For more details on the evolution of the loss use tensorboard to open `data/cnn-runs`
- The console output visualizes the progress of the training

### Examples

To finetune (update all the weights) a model for at most 30 epochs on the HQ dataset, 
with early stopping set to 5 epochs, use the following command:
```shell
python classify_exception.py -d HQ -l -e 30 --early_stopping 5 -bs 32
```
To test the accuracy of a pretrained net, a model path must be given:
```shell
python classify_exception.py -d HQ -mi ../data/models/xception_HQ_e6_finetuning
```

### Replicate all our experiments

Populate the `proj-4/data` folder with all the datasets. Run the `proj-4/classification/run_all.sh` script. 
You may need to adjust the batch size, depending on the amount of VRAM available. 

## Requirements
 
- python 3.7
- `pytorch==1.9.0+cu102` with `torchvision==0.10.0+cu102`
- requirements.txt
