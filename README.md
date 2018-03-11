## Vehicle Detection and Tracking
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/Capture.JPG
[image2]: ./examples/demo.gif
[image3]: ./examples/Capture2.JPG
[image4]: ./examples/Capture3.JPG
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---

### How To run

Please review the installation instructions [here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/README.md)

---

![alt text][image2]

---

### Drawing boxes

I started by writing a function that takes as arguments an image and a list of bounding box coordinates for each car. This function then drew bounding boxes on a copy of the image and return that as its output.

![alt text][image1]


### Histogram of Oriented Gradients (HOG)

Histogram of Oriented Gradients (HOG) can be used to detect any kind of objects. As to a computer, an image is a bunch of pixels and we may extract features regardless of their contents. 
Here, I used OpenCV's function `hogDescriptor()`, within a function I called `get_hog_features`, to get the HOG of my images.

![alt text][image3]

### Extracting Features

Here I defined `extract_features` to loop through images, and create an array of HOG features for each image. This array is then used for training 

`def extract_features(imgs, cspace='RGB', size = (64,64)):
    features = []
    hog = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)
    for filename in imgs:
        image = imread(filename)
        if size != (64,64):
            image = cv2.resize(image, size)
        features.append(np.ravel(hog.compute(get_feature_space(image, cspace))))
    return features

vehicle_features = extract_features(vehicles, cspace='YUV')
non_vehicle_features = extract_features(non_vehicles, cspace='YUV')`

`YUV` color space seemed to have the best performance for identifying vehicles and I relied on it for feature extraction.

### Training

For training, I evaluated the performance of `Support Vector Machine` (SVM) and `Multi-layer Perceptron` (MLP), and ended up with these results

|Classifier|Training Accuracy|Test Accuracy|
|----------|-----------------|-------------|
|SVM |1.00|0.95016|
|MLP |1.00|0.9912|

Both classifiers performed well, but `MLP` had a better Test Accuracy, so I used it to identify vehicles in images and in the video stream.

### Finding Cars

The process of finding cars I used is as follow:

* Implement a sliding window
* look at one slice of the image
* Make predictions of HOG features

Because it doesn't make sense to look for cars in the sky (at least nowadays), I limited my search in the bottom half of the image.
For my predictions, I chose a threshold of `0.98` because with it I could annotate vehicles in the 6 test images provided and 2 more I screen shot from the video.

![alt text][image4]


### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

False positive may challenge my pipeline. It can fail detecting motorcycles or bicycles for exemple. And it may also falsely identify non-vehicles like trees as vehicles. To fix these two problems, it is important to improve the training dataset by adding motorcycles, bicycles and negative features
