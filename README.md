# Project for TimeCode video recognition 

### using neural model from 
```
https://github.com/Tony607/keras-image-ocr
```
### in progress

datasets links will be later 

## Dependencies

### install mask_rcnn

git clone https://github.com/matterport/Mask_RCNN.git

cd Mask_RCNN

sudo ~/PycharmProjects/TimeCodeDetection/venv/bin/python3 setup.py install

## Education  

Create image dataset with annotations 

Train Mask_RCNN model on image dataset for detection part 

Train Recognition model via script in train folder 

add detection model name in config.json 

add recognition model name in config.json 