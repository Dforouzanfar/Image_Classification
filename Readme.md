# Binary Image Classification with Convolutional Neural Networks
This repository contains the code and documentation for a binary image classification project using Convolutional Neural Networks (CNNs), developed for the Machine Learning course at the University of Milan.

This project aims to develop and compare three different CNN architectures for binary image classification: TinyVGG, AlexNet, and ResNet.
The models were trained and evaluated using three different learning rates to determine the optimal configuration. The best performance was achieved using a learning rate of 0.001 with AlexNet architecture (Accuracy on the test set: 94.5%).

### Note on checking the performance of each trained model on the test set.
For doing this, you can download them and upload them on the notebook with `google.colab.files.upload()` and perform the test on the test set.
### Instruction:
1. Go to the section 0.4 and change the `loaded_model` value to True. Then upload the trained model.
2. Go to the section 4.1 and change the values of `my_model` and `learning_rate` to the model that you upload in the last step.
3. Run all the cells on section 5.
