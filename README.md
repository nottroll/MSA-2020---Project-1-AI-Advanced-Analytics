
# Predicting Concrete Strength with Artificial Neural Networks
This project is part of Microsoft Student Accelerator 2020: AI & Advanced Analytics and is my first attempt at an artificial neural network. I have chosen to model the strength of concrete as it is not only the material that is the foundation for every structure but is also relevant to my studies in civil engineering.

  ## Summary
  Concrete is one of the most important materials in the world and is ubiquitous in buildings. However, unlike other equally important materials, concrete is highly complex in its composition and formation at the molecular level. Due to this, predicting its strength through physically derived models is particularly difficult and predictions largely rely on experimental data.

This project aims to predict the compressive strength of concrete with an artificial neural network from a dataset provided by Yeh, 1998. Through the guidance of Yeh and others [<sup>[1]</sup>](#References), a model for regression is built with TensorFlow's implementation of Keras with 8 input parameters, X hidden layers and 1 output layer.

## Environment Setup
This project uses TensorFlow 2.2.0 and Keras Tuner 1.0.1. Instructions for installing TensorFlow locally can be found [here](https://www.tensorflow.org/install). This project also benefits from GPU compute.

    import tensorflow as tf
    import kerastuner as kt
    
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras import regularizers
    
    from kerastuner import HyperModel, RandomSearch, BayesianOptimization, Hyperband
    
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    print('Running TensorFlow  ', tf.__version__)
    print('Running KerasTuner  ', kt.__version__)
    print("GPUs Available:  ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    import pandas as pd
    import numpy as np
    
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    import seaborn as sns

## What I Learnt

 - TensorFlow and Keras are very beginner friendly and have abundant resources for learning.
 - Resources that were particularly useful include Kaggle, Microsoft Learn, TensorFlow Tutorials, 
François Chollet, Weights and Biases
 - Overfitting 
 - Importance of tuning hyperparameters
 
## Results
The model was able to predict X

## References
I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998).
