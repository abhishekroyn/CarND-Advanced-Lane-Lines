## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration2.jpg "Original Test Image Calibration"
[image2]: ./output_images/calibration2_corners_found6.jpg "Test Image with Corners Highlighted"
[image3]: ./output_images/calibration2_undistored_image.jpg "Undistored Test Image Calibration"
[image4]: ./output_images/test1_original.jpg "Original Test Image"
[image5]: ./output_images/test1_undistorted.jpg "Undistored Test Image"
[image6]: ./output_images/test1_preprocessed.jpg "Preprocessed Test Image"
[image7]: ./output_images/test1_undistored_rgb_image.jpg "Undistorted RGB Test Image"
[image8]: ./output_images/test1_warped_cropped.jpg "Warped Cropped Test Image"
[image9]: ./output_images/test1_result_raw.jpg "Result Raw Test Image"
[image10]: ./output_images/test1_result.jpg "Result Test Image"
[video1]: ./output1_tracked.mp4 "Output Video for project_video.mp4"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell under 'Helper Functions' of the IPython notebook located at "./advanced_lane_lines_pipeline.ipynb" in lines 1 through 40.  

I started by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

##### Original Test Image Calibration
![alt text][image1]

##### Test Image with Corners Highlighted
![alt text][image2]

##### Undistored Test Image Calibration
![alt text][image3]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the second code cell under 'Helper Functions' of the IPython notebook located at "./advanced_lane_lines_pipeline.ipynb" in line 75.

I applied the camera calibration and the distortion coefficients (which was computed and saved earlier using `pickle`) to the test image using the `cv2.undistort()` function and obtained this result: 

##### Original Test Image
![alt text][image4]

##### Undistored Test Image
![alt text][image5]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the second code cell under 'Helper Functions' of the IPython notebook located at "./advanced_lane_lines_pipeline.ipynb" in lines 42 through 86.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in line 53 for x or y gradient; in line 60 for saturation; in line 65 for value; in line 68 for saturation and value together; and in line 85 for x gradient, y gradient and color thresheolds all together.  Here's an example of my output for this step.

##### Preprocessed Test Image
![alt text][image6]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is contained in the second code cell under 'Helper Functions' of the IPython notebook located at "./advanced_lane_lines_pipeline.ipynb" in lines 88 through 109.

This section includes a function called `find_perspective_transform()`, which takes as inputs a preprocessed binary image (`preprocessImage`), image size (`img_size`), as well as source (`src`) and destination (`dst`) points, and generates a perspective tranformed image (as bird-view) using OpenCV functions, `cv2.getPerspectiveTransform` and `cv2.warpPerspective`.  I chose to define the source and destination points in a function called `find_src_dst_roi_vertices()`, which calculates source (`src`) and destination (`dst`) points in lines 90 through 92 in the following manner:

```python
src = np.float32(
    [[img_size[0]*(.5-mid_width/2), img_size[1]*height_pct], 
    [img_size[0]*(.5+mid_width/2), img_size[1]*height_pct], 
    [img_size[0]*(.5+bot_width/2), img_size[1]*bottom_trim],
    [img_size[0]*(.5-bot_width/2), img_size[1]*bottom_trim]])
offset = img_size[0]*.25
dst = np.float32(
    [[offset, 0], 
    [img_size[0]-offset, 0], 
    [img_size[0]-offset, img_size[1]], 
    [offset, img_size[1]]])
```
Based on trial and error, the heuristic values for width and height percentage was chosen to be,

| parameters    | values        | 
|:-------------:|:-------------:| 
| mid_width     | .08           | 
| bot_width     | .76           |
| height_pct    | .62           |
| bottom_trim   | .935          |

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 588, 446      | 320, 0        | 
| 691, 446      | 960, 0        |
| 1126, 673     | 960, 720      |
| 153, 673      | 320, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

##### Undistorted RGB Test Image
![alt text][image7]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for this step is contained in the second code cell under 'Helper Functions' of the IPython notebook located at "./advanced_lane_lines_pipeline.ipynb" in lines 111 through 238.

This section includes several functions for specific tasks,

`get_roi()` : I created a masked image where only a section of the image from both left and right sides were masked to suppress noise in the warped image. The roi_vertices as defined in lines 94 to 101,

```python
roi_left_padding = img_size[0]*.20
roi_right_padding = img_size[0]*.15
roi_vertices =  np.array([
                    [roi_left_padding, 0],
                    [img_size[0] - roi_right_padding, 0],
                    [img_size[0] - roi_right_padding, img_size[1]], 
                    [roi_left_padding, img_size[1]]]
                    , dtype=np.int32)
```
This resulted in the following region of interest vertices:

| roi_vertices  | 
|:-------------:|
| 256, 0        |
| 1088, 0       |
| 1088, 720     | 
| 256, 720      |

Here's an example of my output for this step.

##### Warped Cropped Test Image
![alt text][image8]

`find_window_centroids()` : Initial center point for the lane-lines was detected based on convolution between a convolution-window and lower quarter section of the warped image. The image was vertically divided into 9 sections. Moving forward, I used convolution within a defined horizontal margin around last center-point, to look for next center point for each of the rest 8 vertical sections in the image. If any of the 8 vertical sections in the image was found to have number of maximum pixels less than a defined threshold `minpix` then for that section the x-coordinate of that center-point remains the same as of the last section. This ensures to eliminate noise in the image around lane-lines and only those regions contritbute in selection of center-points which have high confidence score to be part of lane-lines. All the center-points for 9 vertical sections in the images was saved along with previous center-points for previously observed frames and an average value of center-points is returned based on `smooth_factor` which helps to display the lane-lines in images continuoulsy and less wobbly. This function is defined in lines 129 through 176 with parameter values as,

| parameters    | values        | 
|:-------------:|:-------------:| 
| window_width  | 25            | 
| window_height | 80            |
| margin        | 50            |
| minpix        | 50000         |
| smooth_factor | 15            |

`window_mask()` : In line 180, this funtion generates a rectangular mask around center-points identified for each of the 9 vertical sections in warped image for both the lane-lines. These rectangular blocks of masks together represents left and right lane-lines in the image. 

`find_lane_mask_pixels()` : In this function, I iterate all the center-points identified so far in the image and highlight pixels around those center points by calling `window_mask()` function. This gives an image with all pixels potentially representing as lane-lines highlighted, whereas rest of all the pixels in the image are suppressed to give a binary image.

`draw_results_left_right_window()` :  This function draws the identified rectangular blocks representing lane-lines on top of the warped image. Here's an example of my output for this step.

##### Result Raw Test Image
![alt text][image9]

`find_lane_boundaries()` : Using `numpy.polyfit()` function on horizontal and vertical coordinates of center-points, I fit my lane lines with a 2nd order polynomial and calculated lane-lines coefficients in lines 226 and 227. Using these coefficient values and for each vertical coordinate value on the image, left and right lane-lines curves were calculated in lines 229 through 233. Finally, I traced the inner and outer coordinates for left-lane and listed them together in line 235. And the same was done for right and inner lane in the lines 236 and 237 respectively.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for this step is contained in the second code cell under 'Helper Functions' of the IPython notebook located at "./advanced_lane_lines_pipeline.ipynb" in lines 256 through 271.

Using `numpy.polyfit()` function on horizontal and vertical coordinates of center-points, I fit my lane lines with a 2nd order polynomial and calculated lane-lines coefficients in lines 258 and 262, but instead of pixels we calculated the values in meters using `ym_per_pix` and `xm_per_pix` values. Thereafter, with the obtained coefficient values and using the equation for calculation of radius of curvature, I computed radius of curvature in meters for both left and right lanes in lines 259 and 263.
`camera_center` was calculated as mid point of left and right lanes at any position on the image. If this `camera_center` value was found to be greater than mid-point of the image on horizontal axis, it resulted in the car having left-offset on the road and it will need to move towards right to be aligned with the lane-center, and vice-versa.
Everything here was calculated or eventualy converted into meters. The `ym_per_pix` and `xm_per_pix` values used here are,

| parameters    | values        | 
|:-------------:|:-------------:| 
| ym_per_pix    | 10/720        | 
| xm_per_pix    | 4/384         |

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for this step is contained in the second code cell under 'Helper Functions' of the IPython notebook located at "./advanced_lane_lines_pipeline.ipynb" in lines 240 through 254. 
Here in the function `draw_results_left_right_inner_lines()`, I used `cv2.fillPoly()` to draw left, right and inner lanes. Thereafter, I used `cv2.warpPerspective()` on the template images with lane-lines drawn to convert them into the format of the original image from camera in lines 249 and 250. 

I added text stating radius of curvature and offset of the car from center in lines 273 through 277 in my code in the same code cell and in the function `draw_curvature_offset()`.  Here is an example of my result on a test image:

##### Result Test Image
![alt text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The code pipeline for this step is contained in the ninth and fiteenth code cells under 'Build an Advanced Lane Finding Pipeline' of the IPython notebook located at "./advanced_lane_lines_pipeline.ipynb".

Here's a [link to my video result](./output1_tracked.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Things I followed,
1) In few frames, this method was not able to detect any hot pixels, so I tried to assign center for that section as `mean of the past center values`. It failed and results were poor. I tried to assign `last center value` in those cases as well. It still failed and resuls were poor.
2) I tried to compare the detected window-center value for a section with previous values and check it's deviation from mean values of past-centers. And if found above a threshold then lower assign a relatively lower value to keep the deviation of the detected window-center closer to the mean value, but that resulted in poor results. I think it may work well when perspective warped image of lane-lines are highly vertical, else it mostly misses the values on curves even if those are genuine values.
3) Thus, I placed the logic to count the number of highlighted pixels in the specific window-section for each of the lanes, and if max value of the convolution of the sliding window with that rectangular section is below a threshold, then window-center of that section is assigned same as window-center of previous section. This logic helped to avoid unnecessary drifitng of lane-lines detection due to noises and gave satisfactory result. So basically, if total pixels > minpix pixels, recenter next window on their highest convolution-value position, else center value remains same as the last value.
4) I tried to improve pre-processing of the image using magnitude and direction gradient with varying values of sobel_kernel and thresholds, but did not get better result than current.
5) I tried to improve pre-processing of the image using more color filters with varying values of thresholds especially using a combination of saturation and lightness, but did not get better result than current.

Things I intend to try in future to improve results for optional harder challengs as my technique did not do well on them,
*   updating src & dst corner points dynamically instead of keeping them static
*   ym_per_pxl and xm_per_pxl may need further adjustment based on trapezoid
*   estimated speed could be shown on video
*   partial window with just the lane lines identification on black background could be shown at top left in video frame
*   handling noises due to shadow on the road with better pre-processing of color filtering
*   handling noises due to road-dividers running along the lanes with better pre-processing of color filtering
*   handling very sharp curves with using approach that once it is known where the lines are in one frame of video, a highly targeted search could be performed for them in the next frame. It could help to track the lanes with sharp-curves better
*   improve selection of `region of interest` vertices to mask out unwanted clutter & noises from binary images
*   improve selection of source `src` and destination `dst` vertices
*   improve selection of window sizes, margin, color and gradient thresholds

