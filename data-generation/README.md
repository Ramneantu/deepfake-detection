# Data generation

## Faceforensics

To generate new data from the FaceForensics database, 
we included their script for downloading the videos, `download_faceforensics.py` and our adaption of the script used for extracting individual frames, `extract_compressed_videos.py`.
First download the desired videos, e.g.:
```shell
python download_faceforensics.py ../data -d Deepfakes -c c40 -t videos -n 100
```
This downloads 100 Deepfake videos with high compression. 
Then you can extract frames from these videos using:
```shell
python extract_compressed_videos.py --data_path ../data -d Deepfakes -c c40 -lv 10
```
which extracts one random frame from the first 10 Deepfake videos.
For more information on the available options, use `python download_faceforensics.py --help` and  
`python extract_compressed_videos.py --help`.
The last step is to arrange the images in the appropriate folder structure. A dataset should contain 3 subfolders, 
`train`, `test` and `val`. Each of those 3 should contain a `real` and `fake` subfolder, where the images are stored.