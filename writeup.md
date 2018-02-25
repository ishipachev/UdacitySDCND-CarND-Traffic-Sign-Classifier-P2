# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[sign1]: ./writeup_pics/setsign1.png
[classes]: ./writeup_pics/elements_per_class_plot.png
[before_after]: ./writeup_pics/before_after.png
[custom_pics]: ./writeup_pics/custom_pics.png
[custom_pics_results]: ./writeup_pics/custom_pics_results.png
[vis_network]: ./writeup_pics/vis_network.png
[vis_network2]: ./writeup_pics/vis_network2.png
[vis_network3]: ./writeup_pics/vis_network3.png
[vis_network4]: ./writeup_pics/vis_network4.png
[vis_network6]: ./writeup_pics/vis_network6.png
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ishipachev/UdacitySDCND-CarND-Traffic-Sign-Classifier-P2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set:

![alt text][sign1]:

It is a bar chart showing how the data spread over classed:

![alt_text][classes]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to fit closed interval [0; 1]. I've calculated min with followed by substraction and division by maximum. At the end of procedure all pictures lay down in range [0...1] with some channel pixels achives 0 value and 1 value. This procedure should help to improve contrast on light or dark colored pictures but keep color channels proportions close to original picture

Here is an example of a traffic sign images before and after normalization:

![alt text][before_after]

I didn't use grayscale convertion becouse I do believe colors should help to detect sign features. If we use it for our own sight why should we drop it for machine learning?

Aswell I don't use any artificial data. Only normalization and full color spectrum usage gave me enough information to achieve near 0.95 accuracy on validation set. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 Input RGB image   					| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x6 				    |
| Flatten				| outputs 400									|
| Fully connected		| outputs 120									|
| RELU					|												|
| Dropout               |                                               |
| Fully connected		| outputs 84									|
| RELU					|												|
| Dropout               |                                               |
| Fully connected		| outputs 43 classes							|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamPotimizer with learning rate equals 0.001, batch size equals 512. Training was done through 50 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.992.
* validation set accuracy of 0.951.
* test set accuracy of 0.930

Well know architecture LeNet was chosen as a model for final solution:
* LeNet architecture as a simple and effective tool to clasiffy small pictures to small amount of classes. 
* It suitable for our task to classify signs. LeNet architecture can be used to solve problems with small size single input like 32x32x3 pictures. The architecture ables to extract features even with small and noisy pictures. LeNet shows good result to process geometry features to classify signs at the end of network pipeline.
* LeNet is small but working solution, could be trained really fast due simple architecture. Easy to modificate and experiment. 
* LeNet provided good results with validation accuracy of 0.962 and test set accuracy of 0.942. There is still some ways to improve network potential with data augmentation and alignment of number of examples per class. Implementaion of these techniques could help to achieve better result without any new raw data.
* Epochs size of 100 can be considered as a train limit with no significant improvements with higher values of number of epochs.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][custom_pics]

All pictures looks like easy to clasify target for our network. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here is prediction results printed as a title for each picture:

![alt text][custom_pics_results]

The model was able to correctly guess 8 of the 9 traffic signs, which gives an accuracy of 0.88%. This is good result with taking into account task to detect numbers with the same network which is trying to sort out signs classes even without numbers.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook. 
For the first image, the model is relatively sure that this is a 50km/h speed limit (probability of 0.575), and the image does contain a stop limit, but 50km/h. The top five soft max probabilities for the first image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.986 | Speed limit (50km/h)
|0.014 | Speed limit (30km/h)
|0.000 | Speed limit (70km/h)
|0.000 | Speed limit (20km/h)
|0.000 | Speed limit (60km/h)

For the second image probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000 | General caution
|0.000 | Traffic signals
|0.000 | Pedestrians
|0.000 | Right-of-way at the next intersection
|0.000 | Road narrows on the right

For the third image:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000 | Keep right
|0.000 | Turn left ahead
|0.000 | Go straight or right
|0.000 | Yield
|0.000 | End of all speed and passing limits

Forth:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|1.000 | Stop
|0.000 | No vehicles
|0.000 | Speed limit (60km/h)
|0.000 | Yield
|0.000 | Speed limit (80km/h)

Fifth:

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|0.760 | Traffic signals
|0.237 | General caution
|0.001 | Road work
|0.001 | Pedestrians
|0.000 | Bumpy road


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

![alt text][vis_network]
![alt text][vis_network2]
![alt text][vis_network3]
![alt text][vis_network4]
![alt text][vis_network6]

Our LeNet implementation works with colored pictures. So at first layer we should expect some kind of color-triggering.

Feature map 4 trying to detect white (or white blue) color. Feature map 3 trying to detect red color. Probably there is some hidden geometry features but it's hard to find them just by looking at layer activation pictures. Color features for the first layers seems reasonable way to detect correct signs. My human way to recognize sign would be:

1. Get color for borders / whole sign
2. Get form of the sign
3. Get the values / figures at the center of the sign

Seems like the LeNet network's first layer working through color detection at first