# AutoGrade - the automatic coin grader

![alt text](https://github.com/sgiles13/coin_grader/blob/main/logo.png?raw=true)

## Introduction

A central issue to the valuation and certification of any coin is the assessment of the coin's condition (i.e., its grade). The coin grading industry has long been dominated by third part grading companies, who charge anywhere from $30 - $100 per coin to grade. In addition to the high cost, the human grading process is also time-consuming with turnaround times on the order of months. Enter AutoGrade. The goal of AutoGrade is to be capable of performing deep learning on thousands of coin images to rapidly assess the coin's grade. This project is currently under development. Thus far, a first-of-its-kind automatic coin cropper has been made available. This tool performs the necessary task of cropping the coin image, thereby removing extraneous information prior to being utilized to train a deep learning model. More information is provided below.

## Loading the Environment

The environment is included here as "environment.yml", and can be cloned via "conda env create -f environment.yml", and activated with "conda activate coin_grader". 

## Coin Cropper
To use the coin cropper, simply specify the location of the coin image to be cropped in "edge_test.py". Then execute "python edge_test.py" and the script returns a cropped image with the specified name. The cropping is done through use of the OpenCV package and HoughCircles routine. The HoughCircles routine identifies the edge of the coin, then a mask is applied which crops the image accordingly. An example of the edge identification and cropped image result is provided below.

![alt text](https://github.com/sgiles13/coin_grader/blob/main/circle_test.png?raw=true)
![alt text](https://github.com/sgiles13/coin_grader/blob/main/crop_test3.png?raw=true)
