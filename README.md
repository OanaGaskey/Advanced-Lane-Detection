# Advanced Lane Finding 

Computer Vision algorithm to compute road curvature and lane vehicle offset using OpenCV Image Processing, Camera Calibration, Perspective Transform, Color Masks, Sobels and Polynomial Fit.

![GIF](output_images/advanced_lane_finding.gif)


The *Advanced Lane Finding* project is a step further from [Lane Lines Detection](https://github.com/OanaGaskey/Lane-Lines-Detection) in identifying the geometry of the road ahead.

Using a video recording of highway driving, this project's goal is to compute the radius of the curvature of the road. Curved roads are a more challenging task than straight ones. To correctly compute the curvature, the lane lines need to be identified but on top of that, the images needs to be undistorted. Image transformation is necessary for camera calibration and for perspective transform to obtain a bird's eye view of the road.

This project is implemented in Python and uses OpenCV image processing library. The source code can be found in the `AdvancedLaneFinding.ipynb` Jupyter Notebook file above. 
The starter code for this project is provided by Udacity and can be found [here](https://github.com/udacity/CarND-Advanced-Lane-Lines).



## Camera Calibration

[Optic distortion](https://en.wikipedia.org/wiki/Distortion_(optics)) is a physical phenomenon that occurs in image recording, in which straight lines are projected as slightly curved ones when perceived through camera lenses. The highway driving video is recorded using the front facing camera on the car and the images are distorted. The distortion coefficients are specific to each camera and can be calculated using known geometrical forms. 

Chessboard images captured with the embedded camera are provided in `camera_cal` folder. The advantage of these images is that they have high contrast and known geometry. The images provided present 9 * 6 corners to work with. 

```
# Object points are real world points, here a 3D coordinates matrix is generated
# z coordinates are 0 and x, y are equidistant as it is known that the chessboard is made of identical squares
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
```

Object points are set based on the common understanding that in a chess board pattern, all squares are equal. This implies that object points will have x and y coordinates generated from grid indexes, and z is always 0. The image points represent the corresponding object points found in the image using OpenCV’s function `findChessboardCorners`.  

```
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Find the chessboard corners
nx = 9
ny = 6
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
```

After scanning through all the images, the image point list has enough data to compare against the object points in order to compute camera matrix and distortion coefficients. This leads to an accurate camera matrix and distortion coefficients identification using ‘calibrateCamera’ function.

```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
undist = cv2.undistort(img, mtx, dist, None, mtx)
```

OpenCV `undistort` function is used to transform the images using the camera matrix and distortion coefficients.

![undistorted_chessboard](output_images/undistorted_chessboard.JPG)

The result of the camera calibration technique is visible when comparing these pictures. While on the chessboard picture the distortion is more obvious, on the road picture it's more subtle. Nevertheless,  an undistorted picture would lead to an incorrect road curvature calculation.

![undistorted_road](output_images/undistorted_road.JPG)



##  Perspective Transform from Camera Angle to Bird's Eye View

To calucluate curvature, the ideal perspective is a bird's eye view. This means that the road is perceived from above, instead of at an angle through the vehicle's windshield.

This perspective transform is computed using a straight lane scenario and prior common knowledge that the lane lines are in fact parallel.  Source and destination points are identified directly from the image for the perspective transform.

![ending_points](output_images/ending_points.JPG)

```
#Source points taken from images with straight lane lines, these are to become parallel after the warp transform
src = np.float32([
    (190, 720), # bottom-left corner
    (596, 447), # top-left corner
    (685, 447), # top-right corner
    (1125, 720) # bottom-right corner
])

# Destination points are to be parallel, taking into account the image size
dst = np.float32([
    [offset, img_size[1]],             # bottom-left corner
    [offset, 0],                       # top-left corner
    [img_size[0]-offset, 0],           # top-right corner
    [img_size[0]-offset, img_size[1]]  # bottom-right corner
])

```

OpenCV provides perspective transform functions to calculate the transformation matrix for the images given the source and destination points. Using `warpPerspective` function, the bird's eye view perspective transform is performed.

```
# Calculate the transformation matrix and it's inverse transformation
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(undist, M, img_size)
```

![warp_perspective](output_images/warp_perspective.JPG)



##  Process Binary Thresholded Images 

The objective is to process the image in such a way that the lane line pixels are preserved and easily differentiated from the road. Four transformations are applied and then combined.  

The first transformation takes the `x sobel` on the gray-scaled image. This represents the derivative in the x direction and helps detect lines that tend to be vertical. Only the values above a minimum threshold are kept.

```
# Transform image to gray scale
gray_img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0)
abs_sobelx = np.absolute(sobelx)

# Scale result to 0-255
scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
sx_binary = np.zeros_like(scaled_sobel)

# Keep only derivative values that are in the margin of interest
sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1
``` 

The second transformation selects the white pixels in the gray scaled image. White is defined by values between 200 and 255 which were picked using trial and error on the given pictures. 

```
# Detect pixels that are white in the grayscale image
white_binary = np.zeros_like(gray_img)
white_binary[(gray_img > 200) & (gray_img <= 255)] = 1
```

The third transformation is on the saturation component using the HLS colorspace. This is particularly important to detect yellow lines on light concrete road. 

```
# Convert image to HLS
hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
H = hls[:,:,0]
S = hls[:,:,2]
sat_binary = np.zeros_like(S)

# Detect pixels that have a high saturation value
sat_binary[(S > 90) & (S <= 255)] = 1
```

The fourth transformation is on the hue component with values from 10 to 25, which were identified as corresponding to yellow. 

```
hue_binary =  np.zeros_like(H)

# Detect pixels that are yellow using the hue component
hue_binary[(H > 10) & (H <= 25)] = 1
```


![binary_thr](output_images/binary_thr.JPG) 



## Lane Line Detection Using Histogram

The lane line detection is performed on binary thresholded images that have already been undistorted and warped. Initially a histogram is computed on the image. This means that the pixel values are summed on each column to detect the most probable x position of left and right lane lines.  

```
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)

# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]//2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint
```

Starting with these base positions on the bottom of the image, the sliding window method is applied going upwards searching for line pixels. Lane pixels are considered when the x and y coordinates are within the area defined by the window. When enough pixels are detected to be confident they are part of a line, their average position is computed and kept as starting point for the next upward window.  

```
# Choose the number of sliding windows
nwindows = 9

# Set the width of the windows +/- margin
margin = 100

# Set minimum number of pixels found to recenter window
minpix = 50

# Identify window boundaries in x and y (and right and left)
win_y_low = binary_warped.shape[0] - (window+1)*window_height
win_y_high = binary_warped.shape[0] - window*window_height
win_xleft_low = leftx_current - margin
win_xleft_high = leftx_current + margin
win_xright_low = rightx_current - margin
win_xright_high = rightx_current + margin
      
# Identify the nonzero pixels in x and y within the window #
good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
       
# Append these indices to the lists
left_lane_inds.append(good_left_inds)
right_lane_inds.append(good_right_inds)
```

All these pixels are put together in a list of their x and y coordinates. This is done symmetrically on both lane lines. `leftx`, `lefty`, `rightx`, `righty` pixel positions are returned from the function and afterwards, a second-degree polynomial is fitted on each left and right side to find the best line fit of the selected pixels.

```
# Fit a second order polynomial to each with np.polyfit() ###
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)   
```

Here, the identified left and right line pixels are marked in red and blue respectively. The second degree polynomial is traced on the resulting image.

![fit_poly](output_images/fit_poly.JPG) 



## Detection of Lane Lines Based on Previous Cycle

To speed up the lane line search from one video frame to the next, information from the previous cycle is used. It is more likely that the next image will have lane lines in proximity to the previous lane lines. This is where the polynomial fit for the left line and right line of the previous image are used to define the searching area. 

The sliding window method is still used, but instead of starting with the histogram’s peak points, the search is conducted along the previous lines with a given margin for the window’s width. 

```
# Set the area of search based on activated x-values within the +/- margin of our polynomial function
left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + 
                prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + 
                prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))).nonzero()[0]
right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + 
                prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + 
                prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin))).nonzero()[0]

# Again, extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds]
```

The search returns `leftx`, `lefty`, `rightx`, `righty` pixel coordinates that are fitted with a second degree polynomial function for each left and right side.  


![prev_poly](output_images/prev_poly.JPG)



## Calculate Vehicle Position and Curve Radius

To calculate the radius and the vehicle's position on the road in meters, scaling factors are needed to convert from pixels. The corresponding scaling values are 30 meters to 720 pixels in the y direction and 3.7 meters to 700 pixels in the x dimension. 

A polynomial fit is used to make the conversion. Using the x coordinates of the aligned pixels from the fitted line of each right and left lane line, the conversion factors are applied and polynomial fit is performed on each. 

```
# Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

# Define y-value where we want radius of curvature
# We'll choose the maximum y-value, corresponding to the bottom of the image
y_eval = np.max(ploty)
    
# Calculation of R_curve (radius of curvature)
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
```

The radius of the curvature is calculated using the y point at the bottom of the image. To calculate the vehicle’s position, the polynomial fit in pixels is used to determine the x position of the left and right lane corresponding to the y at the bottom of the image. 

```
# Define conversion in x from pixels space to meters
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# Choose the y value corresponding to the bottom of the image
y_max = binary_warped.shape[0]

# Calculate left and right line positions at the bottom of the image
left_x_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
right_x_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2] 

# Calculate the x position of the center of the lane 
center_lanes_x_pos = (left_x_pos + right_x_pos)//2

# Calculate the deviation between the center of the lane and the center of the picture
# The car is assumed to be placed in the center of the picture
# If the deviation is negative, the car is on the felt hand side of the center of the lane
veh_pos = ((binary_warped.shape[1]//2) - center_lanes_x_pos) * xm_per_pix 
```

The average of these two values gives the position of the center of the lane in the image. If the lane’s center is shifted to the right by `nbp` amount of pixels that means that the car is shifted to the left by `nbp * xm_per_pix meters`. This is based on the assumption that the camera is mounted on the central axis of the vehicle. 



## Video Output

Check out the restulting video! You can download the video here: [project_video_output.mp4](project_video_output.mp4)