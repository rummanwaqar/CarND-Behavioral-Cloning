# Self Driving Car Behavioral Cloning


The goals of this project are the following:
* Use the simulator to collect data of good driving behavior
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

The project includes the following files:
* [model.py](model.py) containing the script to create and train the model
* [dataset_utils.py](dataset_utils.py) containing all data handling utilities including dataset preprocessing and generator
* [drive.py](drive.py) for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network

#### Running the Car
Using the [Udacity provided simulator](https://github.com/udacity/self-driving-car-sim) and my drive.py file, the car can be driven autonomously around the track by executing
```
python drive.py model.h5
```

#### Importing dataset
* The dataset used for training car be imported using `./get_data.sh`
* Image paths can be converted from absolute to relative using `python dataset_utils.py --fix_paths`

#### Viewing the dataset
* You can see a subset of preprocessed images using `python dataset_utils.py --display`
* You can see the dataset distribution using `python dataset_utils.py --dist`

---

[//]: # (Image References)

[model]: ./output_images/model.png "Model Visualization"
[preprocessed]: ./output_images/preprocessing.png "Preprocessing"
[unbalanced]: ./output_images/all_cam.png "Unbalanced dataset"
[balanced]: ./output_images/balanced.png "Balanced dataset"
[loss_graph]: ./output_images/loss_graph.png "Loss graph"
[output]: ./output_images/output.gif "Output GIF"

## Model Architecture and Training Strategy

The model architecture was based on [Nvidia End to End Learning for Self-Driving Cars](http://arxiv.org/abs/1604.07316) paper. The following image shows our model architecture:

![model]

Input images of size 64x64 were used; the images were normalized using the Keras lambda layer. 5x5 filters with stride of 2x2 was used for the first three convolutions. 3x3 filters with stride of 1x1 was used for last two convolutions (model.py lines 23-27).

Three fully connected layers were added to the last convolution layer. Dropout was added to the fully connected layers to help reduce overfitting. (model.py line 30-36). ReLU activation function was used throughout the model.

### Dataset Preprocessing and Augmentation

The data preprocessing pipeline followed these steps:
1. Images were converted from RGB to YUV as recommended in the Nvidia paper. Experiments with both colourspaces also showed that the car performed better in YUV colourspace.
2. The top and bottom of the image is cropped off to remove pixels that didn't contain road information
3. The image is resized to 64x64 pixels. We were experimentally able to determine that resizing that no affect on our output but made training faster.

![preprocessed]

The three camera images (center, left and right) were merged into the dataset by adding a +/- 0.25 correction factor for the steering angle of the left and right images. As seen by the following histogram the distribution of the dataset is not very balanced. We have a lot of samples for -0.25, 0.0 and +0.25 degrees.

![unbalanced]

We employed the following techniques to balance the dataset (code in dataset_utils.py):
1. Collected more data focusing less frequent steering angles.
2. Applied a max limit on frequently occurring steering angles thereby reducing their frequency.
3. Randomly flipped the images randomly to balance the left and right steering angles.
4. Randomly translated images along the x-axis within +/- 100 pixels. A factor of 0.0025 was applied to the steering angle to compensate for this translation.

The following histogram shows the more balanced dataset:
![balanced]

### Training

20% of the dataset was set aside for validation. Mean squared error was used for loss computation. The model used Adam optimizer and the learning rate was set to 1e-4. We used a batch size of 128 and trained the model for 20 epochs.

![loss_graph]

---

## Video
The following gif shows the output of the trained model on a test track:

![output]

The complete output video can be found [here](track1.mp4).
