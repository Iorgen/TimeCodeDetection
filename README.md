# Project for TimeCode video recognition 

### using neural models, i modified that
For Recognition
```
https://github.com/Tony607/keras-image-ocr
```
For detection 
```
https://github.com/lars76/object-localization
```

# Datasets links
### Videos for detection dataset generation 
http://dl.yf.io/bdd-data/bdd100k/video_parts/
ftp://svr-ftp.eng.cam.ac.uk/pub/eccv/

# Dependencies
pip3 install -r requirements
or 
pip install -r requirements

# Education  

Create image dataset with annotations 
Train MobileNetV2 model on image dataset for detection part 
Train Recognition model via script in train folder 
add detection model name in config.json 
add recognition model name in config.json 
put video samples directory "videos" in train folder
run generate_detection_set.py


#  Deployment

#### For linux 
run static script creation using terminal command: ./static

#### For windows
not created yet

### 
ffmpeg video codec should be install on your platform
need install them to windows or to linux

### 
For me 
python3.6 -m pip install -r requirements.txt - package installation ubuntu 16.04 
