# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2019_11_28_01_07_51_225.jpg "Grayscaling"
[image3]: ./examples/recover_left.jpg "Recovery Image"
[image4]: ./examples/recover_left2.jpg "Recovery Image"
[image5]: ./examples/recover_right.jpg "Recovery Image"
[image6]: ./examples/flipped.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: Invidia.jpeg	"Model evaluation error"
[image9]: ./examples/cropped.png	"Model evaluation error"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* training_invidia.py containing the script to train and save the model
* drive.py for driving the car in autonomous mode
* model_invidia.h5 containing a trained convolution neural network 
* Invidia.jpeg mean square error loss of Invidia model
* run3.mp4 A video file driving in autonomous mode using Invidia model
* writeup_report.md or writeup_report.pdf summarizing the results
* InceptionV3.py. It did not perform well, the car kept driving off the road (not part of the project, for revision only)
* model_inceptionV3.h5 containing a trained convolution neural network using InceptionV3.py (not part of the project, for revision only)
* InceptionV3.jpeg mean square error loss of InceptionV3 model (not part of the project, for revision only)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_invidia.h5
```

#### 3. Submission code is usable and readable

The training_invidia.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 and filter sizes and depths between 24 and 64 (training_invidia.py lines 86-101) 

The model includes RELU layers to introduce nonlinearity (code lines 86-101), and the data is normalized in the model using a Keras lambda layer (code line 82).
Not relavant pixels are also cropped out from the top (50 lines) and bottom (20 lines) of each image 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 115 - 126). 
Early stop of the training and the best model was saved using the validation loss as a metric.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (training_invidia.py line 111).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and trained the vehicle enough on curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to add convolutional networks with different sizes and afterwards fully connected layers.
Relu function was used for activation in each conv layer to introduce nonlinearity.

My first step was to use a convolution neural network model similar to the Invidia team I thought this model might be appropriate because they ran it on a similar task, hence it should work.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I ran the training for 10 epochs and got the following results. The model with the least validation loss was chosen to avoid overfitting.

![alt text][image8]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (curves). To improve the driving behavior in these cases, I had to add some more training data on those curves.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (training_invidia.py lines 89-108) consisted of a convolution neural network with the following layers and layer sizes:
* 2D Convolution with a 5x5 filter size with 24 output filters on a 90x320 image
* 2D Convolution with a 5x5 filter size on top, with 36 output filters
* 2D Convolution with a 5x5 filter size on top, with 48 output filters
* 2D Convolution with a 3x3 filter size on top, with 64 output filters
* 2D Convolution with a 3x3 filter size on top, with 64 output filters
* Fully connected layer of the size 100
* Fully connected layer of the size 50
* Fully connected layer of the size 10
* Fully connected layer of the size 1

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct its path when off the center of the road.
These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]
![alt text][image5]


To augment the data sat, I also flipped images and angles thinking that this would help my model generalize For example, here is an image that has then been flipped:

![alt text][image6]



After the collection process, I had X number of data points. I then preprocessed this data by two steps:
* averaging pixel values from the range [0, 255] to the range [-1.0, 1.0]
* cropping the pixels which are not part of the road, here is an example of the output

![alt text][image9]


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by the figure above. I used an adam optimizer so that manually training the learning rate wasn't necessary.
