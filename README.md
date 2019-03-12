# Human Activity Recognition

This module contains code for analysing different datasets of smartphone sensor data as well 
as code for training and testing an LSTM and a CNN on predicting whether the recorded
sensor data was obtained from a running or a walking person. (As part of a challenge for a PhD position).
Goal of the challenge was to build a mobile app, which could classify 10 seconds of movement into 
`walking` or `jogging/running` activity.  

Each network is given a unique specifier. For each successful run, a folder with the 
network's specifier and the current datetime will be created with train and test results + the set of 
used hyperparameters.

### Datasets

1.  ***WISDM*** 
The Wireless Sensor Data Mining (WISDM) Lab has provided two datasets, which 
contain accelerator measurements. I have used the lab data (not the actitracker)

    > - [dataset](http://www.cis.fordham.edu/wisdm)
    > - [paper](http://www.cis.fordham.edu/wisdm/public_files/sensorKDD-2010.pdf)
    > - Sampling rate = 20Hz 
    > - Device = Android
    > - Carried in front pocket
    > - Feature set: downstairs, upstairs, walking, jogging, sitting, and standing

2. ***MotionSense***
    > - [dataset](https://www.kaggle.com/malekzadeh/motionsense-dataset)
    > - [paper](https://arxiv.org/pdf/1802.07802.pdf)
    > - Sampling rate = 50Hz 
    > - Device = iPhone6
    > - Carried in front pocket
    > - Feature set: downstairs, upstairs, walking, jogging, sitting, and standing

3. ***Run/Walk***
    > - [data](https://www.kaggle.com/vmalyi/run-or-walk)
    > - Sampling rate = 50Hz 
    > - Device = iPhone5
    > - Carried on left and right wrist
    > - Feature set: walking, jogging

### File-structure
    
### Predicting Human Activity

### iOS App
