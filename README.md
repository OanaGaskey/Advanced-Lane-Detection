# Advanced Lane Finding 

Computer Vision algorithm to compute road curvature and lane vehicle offset using OpenCV Image Processing, Camera Calibration, Perspective Transform, Color Masks and Sobels.

![GIF](output_images/LaneFinding.gif)


The *Advanced Lane Finding* project is a step further from [Lane Lines Detection](https://github.com/OanaGaskey/Lane-Lines-Detection) in identifying the geometry of the road ahead.

Using a video recording of highway driving, this project's goal is to compute the radius of the curvature of the road. Curved roads are a more challenging task than straight ones. To correctly compute the curvature, the lane lines need to be identified but on top of that, the images needs to be undistorted. Image transformation is necessary for camera calibration and for perspective transform to obtain a bird's eye view of the road.

This project is implemented in Python and uses OpenCV image processing library. The source code can be found in the `AdvancedLaneFinding.ipynb` Jupyter Notebook file above. 
The starter code for this project is provided by Udacity and can be found [here](https://github.com/udacity/CarND-Advanced-Lane-Lines).


## Camera Calibration

[Optic distortion](https://en.wikipedia.org/wiki/Distortion_(optics)) is a physical phenomenon that occurs in image recording, in which straight lines are projected as slightly curved ones when perceived through camera lenses. The highway driving video is recorded using the front facing camera on the car and the images are distorted. The distorsion coefficients are specific to each camera and can be calculated using known geometrical forms. 

Chessboard images captured with the embedded camera are provided in `camera_cal` folder. The advantage of these images is that they have high contrast and known geometry. The images provided present 9 * 6 corners to work with. 

```
# Object points are real world points, here a 3D coordinates matrix is generated
    # z coordinates are 0 and x, y are equidistant as it is known that the chessboard is made of identical squares
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
```

Object points are set based on the world’s knowledge that in a chess board pattern all squares are equal, this implies that object points will have x and y coordinates generated from grid indexes and z is always 0. The image points represent the corresponding object points found in the image using OpenCV’s function ‘findChessboardCorners’.  

```
		# Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
```

After scanning through all the images, the image points list has enough data to compare against the object points in order to compute camera matrix and distortion coefficients. This leads to an accurate camera matrix and distortion coefficients identification using ‘calibrateCamera’ function.

```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
undist = cv2.undistort(img, mtx, dist, None, mtx)
```

OpenCV `undistort` function is used to transform the images using the camera matrix and distortion coefficients.

[undistorted_chessboard](output_images/undistorted_chessboard.JPG)

The result of the camera calibration technique is visible when comparing these pictures. While on the chessboard picture the distorsion is more obvious, for the road picture it's more subtile. Nevertheless,  undistorted pictured would lead to an incorrect road curvature calculation.

[undistorted_road](output_images/undistorted_road.JPG)


##  Perspective Transform from Car Camera to Bird's Eye View

