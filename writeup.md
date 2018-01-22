# **Traffic Sign Recognition** 

## Writeup

### I tried to articulate my thoughts and observations on this exercise of identifying German Traffic signs using convolutional neural networks with LeNet model as baseline.

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

[image10]: ./writeup-resources/training-visual.png "Given Training Dataset Visualization"
[image11]: ./writeup-resources/test-visual.png "Given Test Dataset Visualization"
[image4]: ./images/right-turn.jpg "Traffic Sign 0"
[image5]: ./images/snow.jpg "Traffic Sign 1"
[image6]: ./images/thirty.jpg "Traffic Sign 2"
[image7]: ./images/work.jpg "Traffic Sign 3"
[image8]: ./images/x-entry.jpg "Traffic Sign 4"

[image9]: ./writeup-resources/test0.png "Test Image 1 Top 5 soft max"
[imagea]: ./writeup-resources/test1.png "Test Image 2 Top 5 soft max"
[imageb]: ./writeup-resources/test2.png "Test Image 3 Top 5 soft max"
[imagec]: ./writeup-resources/test3.png "Test Image 4 Top 5 soft max"
[imaged]: ./writeup-resources/test4.png "Test Image 5 Top 5 soft max"



## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup

#### Q 1. Provide a Writeup that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/komerpoodari/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python numpy library to calculate summary statistics of the traffic
signs data set:

* The size of original training set is 34799. approximately 80% of this given set, i.e. 27839 is used as trainig set.
* The size of the validation set is 6960,  approximately 20% of original training set, using 'train_test_split()'.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3), i.e.32 x 32 with, 3 color channels.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the given data sets. The  following is a bar chart of how training data is distributed among 43 unique  classes.

![alt text][image10]


The  following is a bar chart indicating the distribution of test example among 43 unique traffic sign classes.
![alt text][image11]

The two important points can be noticed from the data set.
1. The distribution of training and test sets are not uniform across traffic sign classes. 
   The classes 0 (20 kmph) and 27 (Pedastrains) have relatively few examples.
   The classes 2 (30 kmph) and 38 (Keep right) have relatively more examples.

2. The distributions of given training and test data sets are similar, which is a good thing consistency perspective.

I had an opportunity to smoothen the distributions by augmenting the data sets. I may work on this aspect later.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize the image pixel values between 0.1 and 0.9.  I did not want to convert to grayscale as I thought color plays a role, for example, stop signs are mostly in red.  The normalization make the distribution pixel values normal distribution offers stability during training.

I will attempt data augmentation later.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input   	    		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling  	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x64  	|
| RELU					|												|
| Max pooling  	      	| 2x2 stride,  outputs 5x5x64   				|
| Fully connected		| Inputs: 1600; nodes: 128						|
| Fully connected		| Inputs: 128; nodes: 64						|
| Fully connected		| Inputs: 64; outputs (logits): 43 classes		|
|						|												|
 
#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The following are final hyper parameters values I used.

| Hyper Parameter  		|     Value  		| Description 										|
|:---------------------:|:-----------------:|:-------------------------------------------------:|
| EPOCHS  	    		| 20  				| I experimented between 10 and 32; selected 20 	|
| LEARN_RATE        	| 0.001         	| played between 0.01 and 0.001; selected 0.001 	|
| BATCH_SIZE        	| 128           	| played between 32 and 256; selected 128       	|
| KEEP_PROB          	| 0.70           	| played between 0.65 and 0.9; selected 0.7     	| 
|						|					|													|

With 20 Epochs I am getting reasonably high validation accuracy > 0.99.
Batch size 128 is reasonable choice between convergence speed and accuracy.
I used AdamOptimizer as it combines the advantages of RMS propagation and Adaptive Gradient optimizations.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

1. First started with a model WITHOUT dropout and with feature sets of conv1 layer 16 and conv2 layer 32 and Epochs 10
2. I could get around 0.92 validation accuracy.
3. I thought the model needed to be a bit complex and increase feature sets of conv1 layer 32 and conv2 layer to 64. Some improvement observed.
4. Dropout with  keep probability increased validation accuracy to > 0.99 and also improved test accuracy to 0.95

My final model results were:
* validation set accuracy of 0.994
* test set accuracy of 0.95

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The model correctly classified 4 images out of 5, with single dominant soft max probability.  One image got classified incorrectly and interesting there are multiple significant soft max probabilities. The image 'Snow' got misclassified, as this category has relatively less number of training and validation examples.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image					|     Prediction			| 
|:---------------------:|:-------------------------:| 
| Right turn      		| Right turn 				| 
| **Beware of ice/snow**| **Speed limit (30km/h)** 	|
| 30 kmph				| 30 kmph					|
| Work	      	    	| Work 	 					|
| No entry		    	| No entry 					|
|						|							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the section, '**Predict the Sign Type for Each Image**' of the Ipython notebook.

For the **first image**, the model is relatively sure that this is a **Right-turn** sign (probability of 0.9999), and the image does contain a **Right-turn** sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| **.9999**    			| **Right-turn**								| 
| 3.76873031e-05		| Turn left ahead 								|
| 7.00325486e-08		| Dangerous curve to the right					|
| 5.13961531e-08		| Go straight or right							|
| 1.40242857e-08		| Slippery road      							|
|						|											 	|

![alt text][image9]

For the **second image** the model predicted not so confidently that this is a **Speed limit (30km/h)** sign (probability of 0.18010029), and the image contains a **Beware of ice/snow** sign.  The top five soft max probabilities computed were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| **0.18010029**		| **Speed limit (30km/h)** 						| 
| 0.17354622			| Right-of-way at the next intersection 		|
| 0.11399166 			| Speed limit (100km/h)							|
| 0.08116101			| Speed limit (80km/h)							|
| 0.05233131			| Speed limit (50km/h) 							|
|						|											 	|

![alt text][imagea]

For the **third image** the model predicted confidently that this is a **Speed limit (30km/h)** sign (probability of 0.994), and the image does contain a **Speed limit (30km/h)** sign.  The top five soft max probabilities computed were

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| **0.994**				| **Speed limit (30km/h)**						| 
| 0.0018427217			| End of all speed and passing limits 			|
| 0.0016041510 			| Speed limit (50km/h)							|
| 0.0015806552			| Speed limit (20km/h)							|
| 0.00018883716			| End of speed limit (80km/h) 					|
|						|											 	|

![alt text][imageb]

For the **fourth image** the model predicted confidently that this is a **Road work** sign (probability of 1.00), and the image does contain a **Road work** sign.  The top five soft max probabilities computed were

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| **1.00**				| **Road work**									| 
| 2.8848233e-11			| Bicycles crossing 							|
| 8.2106657e-12 		| Wild animals crossing							|
| 5.4689060e-13			| Traffic signals								|
| 4.6975450e-13			| Road narrows on the right 					|
|						|											 	|

![alt text][imagec]

For the **fifth image** the model predicted confidently that this is a **No entry** sign (probability of 0.9999), and the image does contain a **No entry** sign.  The top five soft max probabilities computed were

| Probability         	|     Prediction								| 
|:---------------------:|:---------------------------------------------:| 
| **0.9999**			| **No entry**									| 
| 7.4082173e-08			| Stop 											|
| 1.4247485e-09 		| Children crossing								|
| 2.4342006e-10			| Speed limit (20km/h)							|
| 2.6245414e-12			| Speed limit (30km/h) 							|
|						|											 	|

![alt text][imaged]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

*I did not attempt this, I will take it up some time later.*
