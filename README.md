# camera_z_transition
#### Description
This is a small python package to estimate the camera motion/zoom
on the z axis.

what it is:
- Estimates the camera motion/zoom on the z axis


what it is NOT:

- It does not calcualte optical flow (although you can see example of that in
the example.py)
- It is not a metric estimation. It will not tell you the exact amount
of transition on the z axis in meter/feet
- It does not estimate any other camera motion such az tilling, panning, 
tracking  etc.

#### Install
```
pip install cam-motion-field
```

#### Usage
```python
from camera_z_transition import estimate_z_transition
import numpy as np

# This is the width and height of the image on which you perform the
# optical flow calculation
width = 1080
height = 720

# You should change this to some real optical flow calculation! see example
# in example.py. Both origins and displacements should be a list of 2D vectors.
origins = np.ones((10,2))*(1080//2)+10
displacements = np.ones((10,2))*(1080//2)+20

# Normalize and scale the optical flow first!
origins[:, 0] = (origins[:, 0] / width) - 0.5
origins[:, 1] = (origins[:, 1] / height) - 0.5
displacements[:, 0] = (displacements[:, 0] / width) - 0.5
displacements[:, 1] = (displacements[:, 1] / height) - 0.5

# Estimate transition on the z axis
z = estimate_z_transition(origins, displacements)

print(z)
```

    Parameters for estimate_z_transition()
    ----------
    origins: numpy array,
        Set of origin vectors. Shape must be (nr_of_points, 2). MUST BE ZERO CENTERED AND SCALED!
    displacements: numpy array,
        Set of displacement vectors (coordinates). Shape must be (nr_of_points, 2)
    focal_length
    
#### How it works
Given any point in the image plane (X,Y) we can estimate their new position
on the image plane (X',Y') given a transition on the x axis [[1]](#1).
 
<img src="https://render.githubusercontent.com/render/math?math=X' = f[tan^{-1}\frac{X}{f}](1+\frac{X^2}{f^2})\beta">
<br>
<img src="https://render.githubusercontent.com/render/math?math=Y' = f[tan^{-1}\frac{Y}{f}](1+\frac{Y^2}{f^2})\beta">

Where parameter X and Y are coordinates on the image plane
and <img src="https://render.githubusercontent.com/render/math?math=\beta">
is the transition on the z axis. X' and Y' are the new coordinates
on the image plane given the transition parameter. 
Notice that both equations require a parameter *f* which is the focal lenght
if the camera. Interestingly it does not seem to have a noticeable effect on the
output. It seams that until *f* and (X,Y) has sensible values the
exact value of *f* does not matter. It is important that I do not have any
mathematical proof on this. It is purely come from visually observing
the behaviour of the function with different *f* and (X,Y). You can have a
look here: [link to Plot](https://www.desmos.com/calculator/hpo4u5xdkb)

Given an observed optical flow it is easy to perform a parameter search
for <img src="https://render.githubusercontent.com/render/math?math=\beta">
using the equations above. 

#### Results
Given two consecutive images from a camera:

<img src="https://github.com/stamas02/camera_z_transition/blob/master/data/image_anim.gif" width="400"/>

We first obtain the optical flow using opencv

<img src="https://github.com/stamas02/camera_z_transition/blob/master/data/optical_flow.jpg" width="400"/>

The estimated <img src="https://render.githubusercontent.com/render/math?math=\beta">
parameter value for the images above is 0.053796382332247594

The gif below shows the original optical flow and the one artificially
generated using the estimated <img src="https://render.githubusercontent.com/render/math?math=\beta">
parameter.

<img src="https://github.com/stamas02/camera_z_transition/blob/master/data/op_flow_anim.gif" width="400"/>

#### References
<a id="1">[1]</a> 
Srinivasan, M.V., Venkatesh, S., Hosie, R.: Qualitative estimation of camera motion
parameters from video sequences. Pattern Recognition 30(4), 593–606 (1997)

#### Bib
    @article{Srinivasan at al.,
      title={Qualitative estimation of camera motion parameters from video sequences},
      author={Srinivasan, Mandyam V and Venkatesh, Svetha and Hosie, Robin},
      journal={Pattern Recognition},
      volume={30},
      number={4},
      pages={593--606},
      year={1997},
      publisher={Elsevier}
    }
