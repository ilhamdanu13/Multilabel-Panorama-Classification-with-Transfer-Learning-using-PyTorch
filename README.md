# The Idea of Tranfer Learning

# Multilabel-Panorama-Classification-with-Transfer-Learning-using-PyTorch
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

MultilabelDataset adalah tempat dimana data akan dimuat, strukturnya adalah "folder_data" ->  dipisah menjadi "test", dan "train" dan tidak dipisah perkelas, karena setiap 1 gambar mencakup lebih dari satu label, sedangkan file metadata dalam bentuk file csv. 

# About MobileNet V2
Brief description of MobileNet V2:
### Overall Architecture
![1_5iA55983nBMlQn9f6ICxKg](https://user-images.githubusercontent.com/86812576/170874052-f80451b4-eaff-4d39-8253-44b2d3153565.png)

### ReLu6 Activation Function
![image](https://user-images.githubusercontent.com/86812576/170874589-a6f3ed78-d502-4064-86fa-18895ff3e2bc.png)

ReLU6 is used due to its robustness when used with low-precision computation. (_source: https://towardsdatascience.com/review-mobilenetv2-light-weight-model-image-classification-8febb490e61c_)




MultilabelDataset is the place where the data will be loaded, its structure is "folder_name"/"file_name". do you think we want to shuffle the testloader? actually if it's not shuffled it's okay, but because at the end it will do a sanity check so I want the data randomly.