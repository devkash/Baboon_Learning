# Baboon Learning

## Overview

Despite a recent dramatic rise in the deployment of animal-borne movement sensors, very few studies have attempted to use movement data to classify fine-scale behavioral activity of animals. This project fills the void by exploring behavioral activity of baboons with deep neural networks. The baboons from the dataset were fitted with e-obs biologging collars that collected data from GPS units, tri-axial accelerometeres and magnetometers. The GPS sampled at 1 Hz and the accelerometer and magnetometer sampled at 12 Hz. Data from these collars amounted to 180,000 labeled samples and approximately 200 million unlabeled samples. 

To find the optimal deep neural network for classification of behavioral activity, the performances of Multilayer Perceptron (MLP), Time-Aggregrated Multilayer Perceptron (TA-MLP), Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), and Convolutional LSTM (CNN-LSTM) were compared. The CNN attained the highest performance with an F1 score of 0.75.

The performances of these models are shown below.

## Results


