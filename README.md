# ActionClassification
Using Acceleration and Rotationrate of a Device to estimate if sitting, running, biking or walking

See [Video on Vimeo](https://vimeo.com/mechlabengineering/activityclassification)

![Classification](Classification2.png)

## Support Vector Classfier

Features used:

* difference of absolute acceleration
* FFT amplitude peak of rotation rate

### Accuracy

Trained/Tested with the [PAMAP2 Dataset](http://pamap.org/demo.html) (subject 106) and got following accuracy:
* 98.68% standing
* 97.84% walking
* 94.89% cycling
* 100.0% running


## Live Demo

with [Tinkerforge IMU](http://www.tinkerforge.com/en/doc/Hardware/Bricks/IMU_Brick.html) with `TinkerLiveClassification.py` (change to your UID)

![Tinkerforge IMU](http://www.tinkerforge.com/en/doc/_images/Bricks/brick_imu_tilted_front_350.jpg)
