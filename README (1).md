
![](header.jpg?raw=true "Title")
# Feature Interpretor Using facial data

A brief description of what this project does and who it's for ----->
> Gender Detection

> Age Detection

> Emotion Detection


## Acknowledgements

 - [Gender Detection Using Keras](https://github.com/ChibaniMohamed/cnn_age_gender/blob/main/train.py)
 - [A version of face Detection](https://github.com/ChibaniMohamed/facemask_detection)
 - [Good material on emotion detection](https://arxiv.org/ftp/arxiv/papers/2012/2012.00659.pdf)


## Authors

- [@sebastian-binu](https://github.com/sebastian-binu)


## Documentation

[Keras Package](https://www.tensorflow.org/guide/keras)

[Open CV](https://docs.opencv.org/)

## 🚀 About Me
I'm a Graduate Student at UCD...(more info in CV)


## Lessons Learned

What did you learn while building this project? What challenges did you face and how did you overcome them?

There was a lot of research involoved while building this project. Tensorflow packages made this project a million times easier since CNN is an easy to use and understand neural network present in tensorflow framework. The hardest challenge however was integrating all three classifiers into one single classifer and optimize the effective models performance. 
This was made possible by openCV's online image identification platform. Moreover obtaining a good dataset was challenging since most commercial use datasets were paid and free once were either too large or too small for training and testing purposes.

Extracting google images was my first course of action but it turned out to be disaster since the performance of the overall model turned out to be close to one which is realistically not possible and model was greatly overfitting. This was due to the fact that google dataset only allowed training of a small batch and image filteration occurs whenever we google search.


This is a good resource on CNN by stanford on neural networks to gain insights on the topic :::: ----- 
https://cs231n.github.io/convolutional-networks/

## DATASETS

https://susanqq.github.io/UTKFace/

https://www.kaggle.com/datasets/frabbisw/facial-age




## Roadmap

- Use CNN to train a deep learning model to predict GENDER

- Use CNN to train a deep learning model to predict AGE

- Use CNN to train a deep learning model to predict EMOTION

- Use functions in CV2 to detect faces within an image

- Combine the results and make a single face detection program.


## Libraries Required

- pandas 
- numpy
- matplotlib.pyplot 
- cv2
- os
- zipfile 
- time
- datetime 
- itertools
- sklearn.model_selection 
- sklearn.metrics 
- tensorflow 
- tensorflow.keras.models 
- tensorflow.keras.layers 
- tensorflow.keras.layers 
- tensorflow.keras.callbacks 
    
## Layers and Architecture

    Model: AGE DETECTION
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)             (None, 198, 198, 32)      320       
                                                                    
    average_pooling2d (AverageP  (None, 99, 99, 32)       0         
    ooling2D)                                                       
                                                                    
    conv2d_1 (Conv2D)           (None, 97, 97, 64)        18496     
                                                                    
    average_pooling2d_1 (Averag  (None, 48, 48, 64)       0         
    ePooling2D)                                                     
                                                                    
    conv2d_2 (Conv2D)           (None, 46, 46, 128)       73856     
                                                                    
    average_pooling2d_2 (Averag  (None, 23, 23, 128)      0         
    ePooling2D)                                                     
                                                                    
    conv2d_3 (Conv2D)           (None, 21, 21, 256)       295168    
                                                                    
    average_pooling2d_3 (Averag  (None, 10, 10, 256)      0         
    ePooling2D)                                                     
                                                                    
    global_average_pooling2d (G  (None, 256)              0         
    lobalAveragePooling2D)                                          
                                                                    
    dense (Dense)               (None, 132)               33924     
                                                                    
    dense_1 (Dense)             (None, 7)                 931       
                                                                    
    =================================================================
    Total params: 422,695
    Trainable params: 422,695
    Non-trainable params: 0
    _________________________________________________________________





    Model: BINARY-GENDER CLASSIFICATION
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)        [(None, 100, 100, 1)]     0         
                                                                    
    conv2d (Conv2D)             (None, 100, 100, 32)      320       
                                                                    
    dropout (Dropout)           (None, 100, 100, 32)      0         
                                                                    
    activation (Activation)     (None, 100, 100, 32)      0         
                                                                    
    max_pooling2d (MaxPooling2D  (None, 50, 50, 32)       0         
    )                                                               
                                                                    
    conv2d_1 (Conv2D)           (None, 50, 50, 64)        18496     
                                                                    
    dropout_1 (Dropout)         (None, 50, 50, 64)        0         
                                                                    
    activation_1 (Activation)   (None, 50, 50, 64)        0         
                                                                    
    max_pooling2d_1 (MaxPooling  (None, 25, 25, 64)       0         
    2D)                                                             
                                                                    
    conv2d_2 (Conv2D)           (None, 25, 25, 128)       73856     
                                                                    
    dropout_2 (Dropout)         (None, 25, 25, 128)       0         
                                                                    
    activation_2 (Activation)   (None, 25, 25, 128)       0         
                                                                    
    max_pooling2d_2 (MaxPooling  (None, 12, 12, 128)      0         
    2D)                                                             
                                                                    
    conv2d_3 (Conv2D)           (None, 12, 12, 256)       295168    
                                                                    
    dropout_3 (Dropout)         (None, 12, 12, 256)       0         
                                                                    
    activation_3 (Activation)   (None, 12, 12, 256)       0         
                                                                    
    max_pooling2d_3 (MaxPooling  (None, 6, 6, 256)        0         
    2D)                                                             
                                                                    
    flatten (Flatten)           (None, 9216)              0         
                                                                    
    dense (Dense)               (None, 128)               1179776   
                                                                    
    dropout_4 (Dropout)         (None, 128)               0         
                                                                    
    dense_1 (Dense)             (None, 2)                 258       
                                                                    
    =================================================================
    Total params: 1,567,874
    Trainable params: 1,567,874
    Non-trainable params: 0
    _________________________________________________________________





        Model: EMOTION DETECTION
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)        [(None, 48, 48, 1)]       0         
                                                                    
    conv2d (Conv2D)             (None, 48, 48, 32)        320       
                                                                    
    dropout (Dropout)           (None, 48, 48, 32)        0         
                                                                    
    activation (Activation)     (None, 48, 48, 32)        0         
                                                                    
    max_pooling2d (MaxPooling2D  (None, 24, 24, 32)       0         
    )                                                               
                                                                    
    conv2d_1 (Conv2D)           (None, 24, 24, 64)        18496     
                                                                    
    dropout_1 (Dropout)         (None, 24, 24, 64)        0         
                                                                    
    activation_1 (Activation)   (None, 24, 24, 64)        0         
                                                                    
    max_pooling2d_1 (MaxPooling  (None, 12, 12, 64)       0         
    2D)                                                             
                                                                    
    conv2d_2 (Conv2D)           (None, 12, 12, 128)       73856     
                                                                    
    dropout_2 (Dropout)         (None, 12, 12, 128)       0         
                                                                    
    activation_2 (Activation)   (None, 12, 12, 128)       0         
                                                                    
    max_pooling2d_2 (MaxPooling  (None, 6, 6, 128)        0         
    2D)                                                             
                                                                    
    conv2d_3 (Conv2D)           (None, 6, 6, 256)         295168    
                                                                    
    dropout_3 (Dropout)         (None, 6, 6, 256)         0         
                                                                    
    activation_3 (Activation)   (None, 6, 6, 256)         0         
                                                                    
    max_pooling2d_3 (MaxPooling  (None, 3, 3, 256)        0         
    2D)                                                             
                                                                    
    flatten (Flatten)           (None, 2304)              0         
                                                                    
    dense (Dense)               (None, 128)               295040    
                                                                    
    dropout_4 (Dropout)         (None, 128)               0         
                                                                    
    dense_1 (Dense)             (None, 3)                 387       
                                                                    
    =================================================================
    Total params: 683,267
    Trainable params: 683,267
    Non-trainable params: 0
    _________________________________________________________________
## Screenshots

![](face%20detected.JPG?raw=true "Title")

Detection on my application is limited to five faces. Code can be tweaked to detect more faces at a time but this will increase the time complexity during training.
## Performance Metrics

- Confusion Matrix for Age Classification

![](normalized%20confusion%20matrix.JPG?raw=true "Title")


- Loss and Accuracy over an extended period of time for each of the three predictions in the model

![](loss%20and%20accuracy.JPG?raw=true "Title")


![](gender%20class%20loss%20and.JPG?raw=true "Title")

![](emotion%20loss%20and%20a?raw=true "Title")
## Future Applications 

Can be combined with surveillance and implemented in shops or other commercial places for an improved security.

![](cctv%20future.jpg?raw=true "Title")



## References

- Deep Face Recognition" by Y. Taigman, M. Yang, M. Ranzato, and L. Wolf This paper presents a deep learning approach for face recognition, which is often a key component of gender recognition systems. Paper Link: https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf


- Analyzing Gender Inequality through Large-scale Facebook Advertising Data" by A. Mislove, T. Sumner, and J. B. Pujol This research explores gender-targeted advertising and could be relevant to understanding the implications of gender recognition technologies. Paper Link: https://arxiv.org/abs/1612.08586
