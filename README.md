# The Idea of Tranfer Learning
## History
  Starting from ILSVRC (ImageNet Large Scale Visual Recognition Challenge)
There is a competition held by ImageNet, regarding image classification, object detection to image segmentation on a large scale, namely:
- Object classification and localization: 150000 images, 1000 categories.
- Object detection: 200 categories.
- Object detection from video: 30 categories.

  The goal is to create a model capable of classifying up to 1000 categories.
Each winner will publish the model they made. So that people can use the model they created, this is called Transfer Learning, which is using the champion's model that has been trained in up to 1000 categories.

## Use pretrained-model for our own case
For example, there is VGG16, the idea is that VGG16 is a feature extractor so if there is an image it will be a small image, then it is flattened and entered into the neural network to predict.

So if you want to use this model in your case, there are 2 phases:
### Phase 1: Adaptation
- Load a pretrained-model.
- Freeze the feature extractor.
- Modify the classifier to our data, leave it unfreezed.
### Phase 2: Fine Tuning
- Unfreeze some/all models to train, but use a lower learning rate. Because if the learning rate is large, it will damage the pretrained model.
- Learning rate use only 10% of that used in the adaptation phase.
- Fine tuning again with a smaller learning rate.

# Multilabel Panorama Classification with Transfer Learning using PyTorch on Google Colab
In this case, we use Multilabel Classification. the difference in multiclass is that the output is only 1 label which must be 100% with logsoftmax, while the multilabel output can be more than 1 label, because each label can have its own probability with Binary Cross Entropy (BCELoss) activation.
# Import Packages
import common packages:

**import numpy as np**

**import matplotlib.pyplot as plt**

import PyTorch's common packages

**import torch**

from **torch** import **nn, optim**

from **jcopdl.callback** import **Callback, set_config**

from **torchvision** import **datasets, transforms**

from **torch.utils.data** import **DataLoader**

**checking for GPU/CPU**

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Dataset & Dataloader
As usual in the Dataset and Dataloader there must be an image folder and a data pipeline that is transform.

For the batch size, 64 will be used, while for the crop size, almost all popular architectures will use 224, so I have to follow that architecture and 224 is a big size for that I need a GPU.

For the architecture will use mobilenet V2, and it has its own rules in the use of the architecture.
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

![Screenshot 2022-05-29 211401](https://user-images.githubusercontent.com/86812576/170873643-18603d46-18c1-4588-99e3-31624220224c.png)

Does our data qualify? yes, our data has 3 channels, the image must also be loaded in the range [0, 1], yes, pytorch has taken care of that, and the interesting thing is that when it trains imagenet data, it does normalization (standard scaling) where the average of RGB is mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. This means that in our colored data we still have to scale according to these rules, because our feature extractor recognizes the color range in a scaled condition. so there will be one additional step where after doing toTensor it must be normalized with the rules above.

# Multilabel Dataset
![multilabel dataset](https://user-images.githubusercontent.com/86812576/170875865-0c0feabf-834f-4439-aabc-97062ca159de.png)

MultilabelDataset is where the data will be loaded, its structure is "folder_data" -> split into "test", and "train" and not separated by class, because each image includes more than one label, while the metadata file is in the form of a csv file.

# About MobileNet V2
Brief description of MobileNet V2:
### Overall Architecture
![1_5iA55983nBMlQn9f6ICxKg](https://user-images.githubusercontent.com/86812576/170874052-f80451b4-eaff-4d39-8253-44b2d3153565.png)
(_source: https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c_)

### ReLu6 Activation Function
![image](https://user-images.githubusercontent.com/86812576/170874589-a6f3ed78-d502-4064-86fa-18895ff3e2bc.png)

ReLU6 is used due to its robustness when used with low-precision computation. (_source: https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c_)

# Architecture and Config
![image](https://user-images.githubusercontent.com/86812576/170876260-886093a3-45b8-4f6b-b437-7ab0d12a5f86.png)

Actually there are many architectures that can be used, but in this case, as described, we will use the MobileNet V2 architecture because this architecture is lighter and more efficient to run.

Then like scikit-learn, we prepare a variable for MobileNet V2, "pretrained=True" to load the model and its weight.

Next, after loading the architecture, freeze all the weights first,

### MobileNet V2 Architecture

In the mnet architecture, after performing the feature extractor, it always ends with a classifier inserted into the neural network. The neural network is even train to data with 1000 classes, because it is train to image net. But we will only train to 5 classes according to our data, the way is to just overwrite and end with sigmoid because our case is a multilabel classification.

The final result of the custom MobileNet V2 architecture for our case is as follows.

![Screenshot 2022-05-29 222857](https://user-images.githubusercontent.com/86812576/170877533-a55cd8f8-261d-45b0-a3a7-6b1d6323f8bc.png)

### Config
Contains the parameters you want to keep when the model is reloaded. In this case I will save the "output_size", "batch_size", and "crop_size".

# Phase 1: Adaption
### MCOC (Model, Criterion, Optimizer, Callback)
![Screenshot 2022-05-30 175652](https://user-images.githubusercontent.com/86812576/170978766-d6b13a81-9ab3-434a-a710-327caed4764f.png)

Where the adaptation phase is to use a standard learning rate, because it's like training on a regular mini neural network, and a little patience to get a benchmark for the first model, but after that, do tuning. It's easy with MCOC (Model, Criterion, Optimizer, and Callback). 

The adaptation phase is also usually faster, and the score will increase with fine tuining, so usually in the transfer learning adaptation phase the score is quite high around 70-80%, once you do fine tuning you can jump to 90%.

# Phase 2: Fine Tuning
Where the learning rate is reduced, and the patience is increased. so the training will probably be slower, because we don't want to spoil what has been learned in the imagenet data. What is done here is:

![Screenshot 2022-05-30 181725](https://user-images.githubusercontent.com/86812576/170981437-8c810a56-5a75-4037-a7ab-9a5bd0024884.png)

Remember when fine tuning remove all or some layers, in this case I remove all. Next, use a lower learning rate optimizer. for callbacks, there is something called reset_early_stop(), because the first model uses early_stop_patience = 2, currently doing this training early stop patience has touched number 2, so reset early stop to 0 and do it again with early stop patience 5.

# Training and Result

![Screenshot 2022-05-30 210428](https://user-images.githubusercontent.com/86812576/171008942-2be53071-37d3-407c-996c-936314f0f5a0.png)

The resulting model is quite overfit. But when checked on the sanitation check the results are quite generalized.

# Predict

![Screenshot 2022-05-30 182604](https://user-images.githubusercontent.com/86812576/170982735-85907fea-8cd2-4473-aa5d-655e4fe88bc4.png)

In the case of multilabel (binary), each label has a probability of 0 - 100%, then using a threshold of 0.5, meaning that the output is greater than 0.5 then it is a prediction (it is in the image) and converts to float32.

### Sanity check predict before train model
Random prediction before train model

![image](https://user-images.githubusercontent.com/86812576/171005368-5b0e8819-1b67-40f0-9d0f-9521673cc75e.png)


### Sanity check predict after train model
![image](https://user-images.githubusercontent.com/86812576/171006350-40e788aa-4ed6-4eb3-a7c6-3b37cc82090c.png)

We can see in the prediction results that the model is quite good because it can predict many images.

If you see the wrong prediction results (red color) the model is actually not too bad. There are several images in the sanity check that are considered wrong predictions, when in fact the model can predict correctly.

# Mislabeled Data?
![image](https://user-images.githubusercontent.com/86812576/171012469-012c07db-3a6b-48e3-8971-49d77970f66d.png)

It turns out that a lot of data is mislabeled. But our model actually managed to predict the images correctly.
