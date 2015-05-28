# ActionClassification
Using Acceleration and Rotationrate of a Device to estimate if sitting, running, biking or walking

See [Video on Vimeo](https://vimeo.com/mechlabengineering/activityclassification)

![Classification](Classification.png)

## Support Vector Classfier

Features used:

* difference of absolute acceleration
* FFT amplitude peak of rotation rate

![SVC](SVClassifier.png)

## Live Demo

with [Tinkerforge IMU](http://www.tinkerforge.com/en/doc/Hardware/Bricks/IMU_Brick.html) with `TinkerLiveClassification.py` (change to your UID)

![Tinkerforge IMU](http://www.tinkerforge.com/en/doc/_images/Bricks/brick_imu_tilted_front_350.jpg)