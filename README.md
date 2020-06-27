
# Predicting Concrete Strength with Artificial Neural Networks
This project is part of Microsoft Student Accelerator 2020: AI & Advanced Analytics and is my first attempt at an artificial neural network. I have chosen to model the strength of concrete as it is not only the material that is the foundation for every structure but is also relevant to my studies in civil and structural engineering.

## Summary
Concrete is one of the most important materials in the world and is ubiquitous in buildings. However, unlike other equally important materials, concrete is highly complex in its composition and formation at the molecular level. Due to this, predicting its strength through physically derived models is particularly difficult and predictions largely rely on experimental data.

This project aims to predict the compressive strength of concrete with an artificial neural network from a dataset provided by Yeh, 1998. Through the guidance of Yeh and others [<sup>[1]</sup>](#References), a model for regression is built with TensorFlow's implementation of Keras with 8 input parameters, 3 hidden layers and 1 output layer.

## Environment Setup
This project uses TensorFlow 2.2.0 and Keras Tuner 1.0.1. Instructions for installing TensorFlow locally can be found [here](https://www.tensorflow.org/install). GPU compute is also used to accelerate training.

### Dependencies
    pip install -U virtualenv
    pip install -U jupyter
    pip install -U Ipython
    pip install -U tensorflow
    pip install -U keras-tuner
    pip install -U pandas
    pip install -U matplotlib
    pip install -U seaborn
    pip install -U scipy

## Running the Model
To run the model, simply run all cells in the notebook sequentially after installing the necessary dependencies with pip.

## What I Learnt

 - TensorFlow and Keras are very beginner friendly and have abundant resources for learning.
 - Resources that were particularly useful include Kaggle, Microsoft Learn, TensorFlow Tutorials, François Chollet, Weights & Biases
 - Overfitting is very prevalent and many factors will lead to it. However, there are also many strategies to prevent overfitting and metrics that can be interpreted to monitor it. 
 - Importance of tuning hyperparameters as they can greatly increase the model performance and tuning can be performed through training loop manually or through automation.
 - Understanding and manipulating the data is vital in model success. Although, I am not sure if I manipulated the data properly to squeeze the most performance out of it.
 - From researching, other machine learning algorithms like random forest seem to be effective on this dataset.
 - Since this dataset had some clustering, it could be beneficial to split dataset into clusters and perform regression on each.
 
## Results
The model was able to predict within 3.88 MAE for the compressive strength of concrete. This is a reasonable result but could be improved.

## References
I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial neural networks," Cement and Concrete Research, Vol. 28, No. 12, pp. 1797-1808 (1998).

https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/

https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network

https://towardsdatascience.com/hyperparameter-tuning-with-keras-tuner-283474fbfbe
