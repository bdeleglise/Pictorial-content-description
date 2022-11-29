# INF8770 Lab3 - Pictorial content description

In this project you will find two algorithms to describe the pictural content of an image
in order to predict it composition. <br> The color histogram is based on statistics on image colors. <br>
The hog is based on image textures thanks to gradient and angle of gradient. 

## Run the algorithm to do the recognition

To run the image recognition algorithm to assign a type to request images thanks to a list
of reference images :

```
python main.py ALGO
```

* ALGO can be -BOB to apply the color histogram algorithm specify by Bob <br>
Or it can be -GRA11 to apply HOG algorithm specify by our group <br>
Be sure to have python and all packages installed. <br>
You can explore other config by personalize parameters define in constant.py file 


## Sources 

* Global : 
    - https://moodle.polymtl.ca/course/view.php?id=1396

* Color Histogram Algorithm :
    - https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_image_histogram_calcHist.php
    - http://www.cambridgeincolour.com/tutorials/histograms1.htm
    - https://medium.com/mlearning-ai/how-to-plot-color-channels-histogram-of-an-image-in-python-using-opencv-40022032e127
    - https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_image_histogram_calcHist.php

* HOG Algorithm
    - https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f#:~:text=Histogram%20of%20Oriented%20Gradients%2C%20also,the%20purpose%20of%20object%20detection.
    - https://lilianweng.github.io/posts/2017-10-29-object-recognition-part-1/
    - https://github.com/trinhngocthuyen/teach-myself-ml/blob/master/funda_ml/HOG.ipynb
    - https://github.com/scikit-image/scikit-image/blob/fe96435877f40581d678b10fde650c6e1899354a/skimage/feature/
    - https://github.com/SamPlvs/Object-detection-via-HOG-SVM/blob/master/testing_HOG_SVM.py