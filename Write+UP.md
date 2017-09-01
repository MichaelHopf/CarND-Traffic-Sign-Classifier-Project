
# **Traffic Sign Recognition**

The goal of the project is to develop a convolutional neural net that recognizes German traffic signs.

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

[image1]: ./examples/hist1.png "Histogram"
[image2]: ./examples/9signs.png "Nine Traffic Signs"
[image3]: ./examples/wrongclass.png "Wrongly Classified"
[image4]: ./examples/ownsigns.png "Own Signs"
[image5]: ./examples/softmax2.png "Softmax"
[image6]: ./examples/conv1.png "Conv 1"
[image7]: ./examples/conv2.png "Conv 2"

# Rubric Points
In the following, I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

## Ipython Notebook

Here is a link to my [project code](https://github.com/MichaelHopf/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)


## Data Set Exploration

### Dataset summary

Basic analysis:
* Training set size: 34799
* Validation set size: 4410
* Test set size: 12630
* Number of classes: 43
* Image data shape: (32,32,3)


### Dataset visualization

The dataset consists of 43 different German traffic signs. However, some signs appear much more often in the training set (or validation set) than others. This can be seen in the following histogram. I did not check the distribution of the signs of the test set to avoid introducing a bias.

![image1]


When viewing some of the training images, one could see that, even for a human, some are quite difficult to recognize, especially due to the low brithness. Here are nine sample images:

![alt text][image2]


## Design and Test a Model Architecture

### Preprocessing
I normalized the images as follows: For each color channel, I calculated the mean over the training set and subtracted it. Then, I divided by 128. I did not do any data augmentation.

### Model Architecture

I started building my net from the basic architecture of LeNet. However, since I did not grayscale the input, I wanted to net to extract important color features itself. Therefore, I used a 1x1 convolution at the beginning with a depth of 10. Then, basically the LeNet architecture follows. However, I substituted the first 5x5 convolution with two 3x3 convolutions of a little greater depth. Also, I introduced three dropout layers in total.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 32x32x10 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 30x30x10 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| Dropout				| 0.5											|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16  				    |
| Flatten    		    | outputs 1x400      							|
| Fully connected		| outputs 1x120     							|
| Dropout				| 0.5											|
| RELU					|												|
| Fully connected		| outputs 1x84     							    |
| Dropout				| 0.5											|
| RELU					|												|
| Fully connected		| outputs 1x43     							    |
| RELU					|												|
| Softmax				| probabilities       							|
 
 

### Model Training

For training, I used the following hyperparameters:
* Learning rate: $5^{-4}$
* Number of epochs: $35$
* Batch size: $128$
* L2 - Regularization: $0$ (no L2 regularization was used)
* Dropout Parameter: $0.5$
* Optimizer: Adam
* Training time: $\approx$ 2 hours (using Microsoft Surface)

In particular, it is worth mentioning that the three dropout layers did a sufficient job of regularizing. Thus, I observed that additional L2 regularization was not beneficial.


### Solution Approach

As already mentioned, I started with the well-known LeNet architecture and added a few more layers. In particular, the three Dropout layers are helpful to prevent overfitting. I added a 1x1 convolution at the beginning to let the net learn what color features are important. Also, I substituted the first 5x5 Convolution of LeNet with two 3x3 Convolutions to make the net a little bit deeper.

After training for 35 epochs, I ended up with the following accuracies:
* Training set accuracy: $99.3\%$
* Validation set accuracy: $96.2\%$
* Test set accuracy: $94.85\%$

This still indicates that the model is overfitting the data. A more extensive hyperparameter search or data augmentation maybe could better the results with the same net. Since I used a Microsoft Surface for training, I could not do an extensive hyperparameter search.

In the following picture, we can see 20 (out of 651) wrongly classified traffic sign from the test set:

![alt text][image3]

One can notice that even if the prediction is wrong, the predicted sign is often similiar, e.g., speed limit signs are often recognized as speed limit signs but with a different speed, or the 'traffic signals' sign is confused with the 'general caution' sign.


## Test a Model on New Images


### Acquiring New Images

Since I live in Germany, I took 35 photos of traffic signs. Some of seem to be quite easily recognizable while others are hard to identify even for a human. Also, I noticed that some traffic signs occur much more frequently than others.


### Performance on New Images

The net did not perform well on my own images with an accuracy of approximately $82\%$. The following picture shows which signs were wrongly classified.

![alt text][image4]


### Model Certainty - Softmax Probabilities

In the following picutre, we can see the softmax probabilities of the last seven traffic signs of my own dataset.

![alt text][image5]



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)


The next picture shows the feature maps of the first 1x1 convolution. We can observe that some neurons react to brigth and some to dark parts.

![alt text][image6]


The next picture shows the feature maps of the first 3x3 convolution. We can observe that shapes become now more important (see also Ipython notebook for further details.

![alt text][image7]


