# KCF tracker in Python

Python implementation of
> [High-Speed Tracking with Kernelized Correlation Filters](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)<br>
> J. F. Henriques, R. Caseiro, P. Martins, J. Batista<br>
> TPAMI 2015

It is translated from [KCFcpp](https://github.com/joaofaro/KCFcpp) (Authors: Joao Faro, Christian Bailer, Joao F. Henriques), a C++ implementation of KCF.

Find more references and code at http://www.robots.ox.ac.uk/~joao/circulant/

### Requirements
- Python 2.7
- NumPy
- Numba (needed if you want to use the hog feature)
- OpenCV (make sure you can `import cv2` in python)

### Usage
```shell
git clone https://github.com/uoip/KCFpy.git
cd KCFpy
python run.py
```
it will open the default camera of your computer, you can also open a different camera, or a video
```shell
python run.py 2
```
```shell
python run.py test.avi  
```
