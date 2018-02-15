# drnet-py
 
Pytorch version of the DrNet project. This code is still a work-in-progress. Scripts for downloading SUNCG and KTH are missing, but everything else should be mostly usable.

I will be msot responsive regarding questions about the code if you email me at denton@cs.nyu.edu.



##  Training on KTH
First download the KTH action recognition dataset by running:
```
sh datasets/download_kth.sh /my/kth/data/path/
```
where /my/kth/data/path/ is the directory the data will be downloaded into. Next, convert the downloaded .avi files into .png's for the data loader. To do this you'll want [ffmpeg](https://ffmpeg.org/) installed. The following script will do the conversion, but beware, it's written in lua:
```
th datasets/convert_kth.lua --dataRoot /my/kth/data/path/ --imageSize 128
```
The ```--imageSize``` flag specifiec the image resolution. Experimental results in the paper used 128x128, but you can also train a model on 64x64 and it will train much faster.
