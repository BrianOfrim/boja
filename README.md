# Boja
An end to end object detection tool. All the way from capturing and labeling a dataset of images to training and deploying an object detection neural network.

This package makes use of that harvesters machine vision image acquisition library for capturing images so a GenICam compliant machine vision camera is required.

Boja translates to "let's see" in Korean  

## Getting Started
### Installing Dependencies
**The following will not work for arm64/aarch64 (Jetson Tx2 etc). For arm64 specific instructions see ARM64 section.**
#### Harvesters
Boja uses the genicam harvesters library for image acquisition.  
Harvesters Repository: https://github.com/genicam/harvesters

Harvesters is a consumer that requires a genTL (GenICam Transport Layer) producer to produce images for it to consume.

A good option for a GenTL producer is one supplied by Matrix Vision. To install it visit:  
http://static.matrix-vision.com/mvIMPACT_Acquire/2.29.0/
And download the following two packages:  
install_mvGenTL_Acquire.sh  
mvGenTL_Acquire-x86_64_ABI2-2.29.0.tgz  

Then run the following with your user (not sudo):
```
$ ./install_mvGenTL_Acquire.sh
```
This will install mvIMPACT_Acquire at the location /opt/mvIMPACT_Acquire  

The file that we are concerned about is the genTL producer which by default will be located at:  
/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti

#### Pip dependencies
**Recommended: Install dependencies inside a virtual env**  

Install numpy and Cython first or else some of the package installation will fail  
```
$ pip install numpy Cython
```
Now install the rest of the dependencies:  
```
$ pip install -r requirements.txt
```

#### ARM64 Installation
The installation process will be different for arm64
**Recommended: Install dependencies inside a virtual env**  

Install numpy and Cython first or else some of the package installation will fail  
```
$ pip install numpy Cython
```
Install some dependencies:  
```
$ pip install -r requirements-arm.txt
```
Install build and install opencv using the instructions from the following page:  
https://pythops.com/post/compile-deeplearning-libraries-for-jetson-nano

You will have to manually copy the resulting cv2 site package into you virtual env site packages if you are using a virtual env

Install pyTorch v1.4 and torchvision v0.5.0 according to the instruction on the following page:  
https://devtalk.nvidia.com/default/topic/1049071/jetson-nano/pytorch-for-jetson-nano/

Since the harvesters library's genicam dependency is not available for arm yet in order to use the capture and predict functionality of boja we will have to use a vendor specific image acquisition library. For FLIR machine vision cameras we can use Spinnaker and PySpin. Included with the boja package there are pySpin variants of the capture and predict modules.
Install the latest Spinnaker **and** PySpin from:  
https://www.flir.ca/products/spinnaker-sdk/


## Usage
### Configure
Boja expects a certain data file structure for file input and output.  
Data file structure:  


├── **local_data_dir**  
│   ├── annotations  
│   ├── images  
│   ├── logs 
│   ├── images  
│   ├── manifests  
│   └── modelstates  

This structure will be created by running the configure module with:  
```
$ python -m vision.configure
```
optional flags include **--local_data_dir** where you can specify where the  
data files will be kept locally. By default this value is:  
```
~/boja/data
```

#### AWS S3 Integration
This project is BYOB (bring your own bucket).  
The configure module, like most other modules in the boja package has the  
optional flag **--s3_bucket_name**. You can use any s3 bucket that you have  
access to. This will allow boja to download preexisting data from the s3  
bucket and also to upload generated data. You can specify the base directory  
of your data within the s3 bucket with the flag **--s3_data_dir**, which by  
default will be:
```
data
```
### Capture

### Label

### Train

### Deploy




