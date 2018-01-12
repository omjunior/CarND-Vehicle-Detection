# Vehicle Detection Project

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a Linear SVM classifier
* Also to apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heatmap of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Training examples)
[ex-car1]: ./writeup/car1.png
[ex-car2]: ./writeup/car2.png
[ex-car3]: ./writeup/car3.png
[ex-car4]: ./writeup/car4.png
[ex-ncar1]: ./writeup/ncar1.png
[ex-ncar2]: ./writeup/ncar2.png
[ex-ncar3]: ./writeup/ncar3.png
[ex-ncar4]: ./writeup/ncar4.png

[//]: # (Frames)
[frame15search]: ./writeup/frame_015_search_2.jpg
[frame151]: ./writeup/frame_015.jpg
[frame152]: ./writeup/frame_015_search.jpg
[frame153]: ./writeup/frame_015_single_heat.jpg
[frame154]: ./writeup/frame_015_acc_heat.jpg
[frame155]: ./writeup/frame_015_labels.jpg
[frame156]: ./writeup/frame_015_output.jpg
[frame161]: ./writeup/frame_016.jpg
[frame162]: ./writeup/frame_016_search.jpg
[frame163]: ./writeup/frame_016_single_heat.jpg
[frame164]: ./writeup/frame_016_acc_heat.jpg
[frame165]: ./writeup/frame_016_labels.jpg
[frame166]: ./writeup/frame_016_output.jpg
[frame171]: ./writeup/frame_017.jpg
[frame172]: ./writeup/frame_017_search.jpg
[frame173]: ./writeup/frame_017_single_heat.jpg
[frame174]: ./writeup/frame_017_acc_heat.jpg
[frame175]: ./writeup/frame_017_labels.jpg
[frame176]: ./writeup/frame_017_output.jpg


## Training the Linear SVC

The labeled data used on this project can be found [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). It consists of 8792 car and 8968 non-car 64 by 64 pixels color images.

Follow some examples of cars and non-cars in this set:

| Cars | Not Cars |
|:---:|:---:|
| ![ex-car1] ![ex-car2] ![ex-car3] ![ex-car4] | ![ex-ncar1] ![ex-ncar2] ![ex-ncar3] ![ex-ncar4] |

After reading the image files, a set of features are extracted using the ```extract_features()``` function, found in ```features.py```. This function will:
* Convert the images to a specified color space (as 'YUV' gave good results, it was the chosen space for this implementation);
* Compute and concatenate features into a features vector.
  * Spatial binning features
  * Color histogram features
  * HOG features

With the features computed, the data is split between the training set and the test set randomly.

The Linear Support Vector Machine is trained varying it's ```C``` parameter from 0.0001 to 1, in steps of roughly half a logarithmic-decade. It was found that the best value, in this case, was 0.003. This implies that a smooth decision boundary is generalizing better for this dataset.

### Spatial Binning

It is simply a down-sampling of the image, unraveling each color channel and concatenating into a single vector. In this project, the images were reduced to an (8, 8) image, and then the 3 channels were unraveled and concatenated.

### Color histogram

For each color channel, a histogram of pixel intensities is computed, aggregated into a vector. I found that 32 bins give good results.

### Histogram of Oriented Gradients (HOG)

In this project, the [scikit-image library](http://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=hog#skimage.feature.hog) was used for HOG computation.
It essentially computes the gradient of an image, resulting in a direction-intensity pair for each pixel. Then the image is divided into cells. Each cell will have a resulting histogram of directions and orientations, which is a composition of the belonging pixels' values. The histograms are later normalized across blocks. The idea is to be a robust feature against small shape variations.

Here the histograms have 11 bins, each cell is 16 by 16 pixels and each block is composed of 2 by 2 cells. These values were found through experimentation and seemed to give an acceptable result.


## Finding Cars

The process of finding cars on an image starts by running a sliding window search to find matches on each image, as implemented in the file ```window.py``` (function ```find_cars()```).
After that a heatmap technique was used in order to remove spurious false positives (functions ```add_heat()``` and ```find_bboxes_from_heatmap()```).


### Sliding Window Search

In order to run a sliding window search, it is necessary first to define the window sizes and the region of interest in the image. In this work, these parameters can be defined on the ```settings.py``` file, by providing minimum and maximum ```x``` and ```y``` coordinates, as well as a multiplier factor. A 1 multiplier defines a window of the same size as the training images (64 by 64 pixels).

The search itself consists of running the classifier for several patches of the image, one at a time, and sliding to the next one. In this project, the windows slide by one-fourth of its size each time.

In the following example, all of the tested positions for one given scale are shown. One of the windows was drawn in red so the size of it becomes clearer.

![frame15search]

The search was performed using the following parameters:

min x | max x | min y | max y | scale
:---:|:---:|:---:|:---:|:---:
100 | 1180 | 400 | 500 | 1
 50 | 1230 | 400 | 550 | 1.5
  0 | 1280 | 380 | 600 | 2
  0 | 1280 | 380 | 650 | 2.5
  0 | 1280 | 380 | 700 | 3

### Heatmap

As the trained SVM finds lots of false positives, a method for filtering them out is necessary.
In this work, a heatmap approach was used.

After the sliding window search is executed, a heatmap is generated for each frame, where the value of each pixel represents the number of positive windows containing that pixel.

As short sequences of frames do not vary much in a video sequence. Hence the heatmap is accumulated over a short sequence of ```n_frames``` frames (the number used here was 13 frames or approximately half a second). In order to penalize non-consistent positives, a geometric mean was used instead of the arithmetic one, so instead of adding the latest ```n_frames```, they are multiplied.
After that, any value equal or below the ```heat_threshold``` value to the power of ```n_frames``` - meaning geometrical mean less or equal to ```heat_threshold``` - is discarded (here a value of 3 was used as the threshold).

## Video Implementation

In order to account for the frame processing in sequence, a ```VideoProcessor``` class was defined in the ```video_processor.py``` file, which runs the whole algorithm over an image frame.

Follow an example of a sequence of frames from a video.

|                         |             |             |             |
|:-----------------------:|:-----------:|:-----------:|:-----------:|
| **Input**               | ![frame151] | ![frame161] | ![frame171] |
| **SVM positives**       | ![frame152] | ![frame162] | ![frame172] |
| **Frame heatmap**       | ![frame153] | ![frame163] | ![frame173] |
| **Accumulated heatmap** | ![frame154] | ![frame164] | ![frame174] |
| **Thresholded**         | ![frame155] | ![frame165] | ![frame175] |
| **Final**               | ![frame156] | ![frame166] | ![frame176] |

### Results

By applying the algorithm to the [input video](./project_video.mp4) with debug settings on, the following [debug video](./output_video/project_video_debug.mp4) was generated. This was the final [output video](./output_video/project_video.mp4).

In the debug version it is possible to see the heatmaps and thresholds together with the output frame.

## Code Structure

The three executable files in this project are:
* ```train_clf.py```: Train a classifier and save the results into pickle files.
* ```search_image.py```: Run the algorithm in single image files, but only looking at 1 frame at a time.
* ```search_video.py```: Run the whole algorithm in a video file, considering the latest frame as defined in the settings.

The supporting files are:
* ```features.py```: Contain functions related to feature extraction.
* ```video_processor.py```: Define a class that can process a frame taking into account previous frames already processed.
* ```window.py```: Contain the functions related to the sliding window search and heatmap.
* ```settings.py```: Establish a centralized place for defining tuning parameters, making sure that the same parameters are used when extracting features both when training the classifier and when running the algorithm over images or videos.

## Discussion

The main drawback of this implementation is the performance of the chosen classifier, which finds many false positives and takes time to run. While it is possible to mitigate the false positives by using some techniques, like the heatmap approach in this work, it would be much better to have a more precise classifier. I strongly suspect that a CNN would perform much better than the linear SVM with HOG features, and classification would run much faster.

Another factor that would make any classifier perform better is to use a time-dependency-aware split between training and test samples. The labeled images provided are strongly correlated - some images are almost identical to the next ones - making the classifier overfit. This can be avoided by carefully grouping these similar images, making sure that the groups are entirely on either the training or test data.

In the end, the threshold value had to be chosen in order not to have too many false positives, and not to lose cars in the prediction. The ultimate result is a balance of these objective, but it is clearly not perfect.
