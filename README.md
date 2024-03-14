# Product Image Classifier

## Overview
This project aims to develop a product image classifier using Convolutional Neural Networks (CNNs). The classifier categorizes images into predefined groups such as fashion, makeup, or accessories.

## Dataset
The dataset consists of images categorized into separate folders for each class: fashion, makeup, and accessories. Each image is resized to 60x80 pixels.

## Training the Model
I used TensorFlow to build and train the CNN model. The model architecture includes convolutional layers, max-pooling layers, and fully connected layers.

### Training Process
the dataset consists of 800 images for accessories class, 1400 images for fashion and 1050 images for the beauty product for the makeup class the dataset then is spilt into training dataset , test dataset and validation dataset with the following percentages train_ratio = 0.7 validation_ratio = 0.15 test_ratio = 0.15
- **Image Preprocessing**: Images Ire rescaled to the range [0, 1] This normalization ensures that the pixel values are on a similar scale, which can help improve the convergence of the optimization algorithm during training and make the model more robust to variations in the input data.
- **Data Augmentation**: Applied to the training data to improve model generalization (data augmentation refers to the process of artificially increasing the diversity of the training dataset by applying various transformations to the existing data samples. These transformations typically include techniques such as rotation, scaling, flipping, cropping, and color manipulation.).
- **Model Training**:  The model was trained using the training dataset with validation performed on a separate validation dataset.

## Evaluation
After training, the model's performance was evaluated on a test dataset. The evaluation metrics include accuracy and loss.
Epoch 1/10
 2s 30ms/step - accuracy: 0.9871 - loss: 0.0388 - val_accuracy: 0.9312 - val_loss: 0.4171
Epoch 2/10
 0s 443us/step - accuracy: 1.0000 - loss: 0.0019 - val_accuracy: 1.0000 - val_loss: 0.0213
Epoch 3/10
 2s 31ms/step - accuracy: 0.9955 - loss: 0.0173 - val_accuracy: 0.9354 - val_loss: 0.3298
Epoch 4/10
 0s 457us/step - accuracy: 0.9688 - loss: 0.0254 - val_accuracy: 1.0000 - val_loss: 1.4236e-04
Epoch 5/10
 2s 30ms/step - accuracy: 0.9969 - loss: 0.0149 - val_accuracy: 0.9312 - val_loss: 0.3802
Epoch 6/10
 0s 501us/step - accuracy: 1.0000 - loss: 5.1846e-04 - val_accuracy: 0.8571 - val_loss: 0.2513
Epoch 7/10
 2s 31ms/step - accuracy: 0.9866 - loss: 0.0316 - val_accuracy: 0.9187 - val_loss: 0.4713
Epoch 8/10
 0s 486us/step - accuracy: 0.9688 - loss: 0.0630 - val_accuracy: 0.8571 - val_loss: 0.1936
Epoch 9/10
 2s 32ms/step - accuracy: 0.9946 - loss: 0.0230 - val_accuracy: 0.9146 - val_loss: 0.5501
Epoch 10/10
 0s 500us/step - accuracy: 0.9688 - loss: 0.0213 - val_accuracy: 1.0000 - val_loss: 7.0693e-04

### Test Accuracy
The model achieved an accuracy of approximately 92% on the test dataset.
Test accuracy: 92.08% this is for the test dataset from the main dataset
![Test 1 Image](results\test_1.jpg)

After testing on the main test data I changed pick some images randomly from slash app and tested the result with
Accuracy: 80.00%
![Test 2 Image](results\test_2.jpg)
## Visualizing Results
I visualized randomly selected images from the test dataset along with their predicted classes. Additionally, I compared the predicted classes with the true labels stored in a CSV file.

## Future Work
- Experiment with different CNN architectures and hyperparameters.
- Explore techniques to handle class imbalance.
- Deploy the model for real-world use, such as in an e-commerce application.

the training dataset link before spliting
(https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

the dataset for the random images to test for the real world images are taken for slash app 
(https://play.google.com/store/apps/details?id=com.slashOrg.slash)
## Contributors
- [Ibraheim Tarek](https://github.com/IbraheimTarek)
