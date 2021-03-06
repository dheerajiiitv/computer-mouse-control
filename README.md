*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Computer Pointer Controller

*TODO:* Write a short introduction to your project.

In this project we will use our eyes and head to control mouse pointer. This project is using multiple models to extract different features. for example head position, eyes location and combining these features to get mouse pointer direction. 

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.

All code files are in the directory 'src/'.
You can see my result in 'output_control.mp4'
Models are located in 'intel/'

## Demo
*TODO:* Explain how to run a basic demo of your model.
*Use below command to execute the program (Default will be cam)*

python3.5 starter/src/main.py -fd starter/intel/face-detection-adas-binary-0001/model/face-detection-adas-binary-0001 -hp starter/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 -ge starter/intel gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002 -fc starter/intel/facial-landmarks-35-adas-0002/FP16/facial-landmarks-35-adas-0002

*precision and speed is default to medium*

## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.

Command line arguments - 
-fd - Location of the face detection model
-hp - Path to head pose estimation model
-ge - Path to gaze estimation model
-fc - Path to facial landmark detection model
-i - Input file
-p" - Precision of mouse controller
-s  - Speed of mouse controller"
-l - MKLDNN (CPU)-targeted custom layers.Absolute path to a shared library with the kernels impl

## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.

hardware 			precision  			Loading time   	 	 	Input/Output Processing time 		Model Inference time
CPU                     FP16               0.97 seconds														4.9 seconds
CPU                     FP32 			   1.15 seconds														5.7 seconds
CPU                     FP32-INT8			4.2 seconds														5.3 seconds

## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.

When we increase float point precision than the model size increases because FP32 take 32 bit of memory while fp16 takes 16 bit of memory, this increase loading time and as floating point increase inference time increases because now mathamatics operation will take more time. 

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.

### Async Inference
*TODO (Optional):* If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.

### Edge Cases
*TODO (Optional):* There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
