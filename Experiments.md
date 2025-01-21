For the colon histology report, we incorporate a comprehensive collection of histology images labeled "ColonHistology" into our Python environment. By using the Google Drive API, we mount Google Drive to access this extensive image collection. Leveraging the os library, known for its powerful file manipulation features, we enumerate the files in the specified directory, allowing us to handle the images systematically.

```python
image_directory = '/content/drive/MyDrive/ColonHistology'
     

files = os.listdir(image_directory)
files
```


Then, we generate a DataFrame from the histology image file names and their respective labels. We start by defining an empty list to store the data and a regular expression pattern to identify labels in the file names. Using the **os** library, we list all files in the specified directory and filter for those with a **.tif** extension. We then match the file names to our label pattern to extract the labels.

With the **pandas** library, we create a DataFrame containing columns for the file paths and their associated labels. To make the labels suitable for machine learning models, we utilize the **LabelEncoder** from the **sklearn** library, converting the string labels into numerical format. Finally, we construct a dictionary to map the encoded labels back to their original string representation and print the classes present in the dataset.

![Code2](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/code2.png)

We split the dataset into training, validation, and test sets to prepare it for machine learning model training and evaluation. We begin by using the **train_test_split** function from the **sklearn** library to divide the data into training/validation and test sets, with the test set comprising 15% of the total data. The split is stratified to ensure that the class distribution remains consistent across the sets.

Next, we further split the training/validation set into separate training and validation sets, with the validation set comprising approximately 5.88% of the total data. Again, we use stratified splitting to maintain the class distribution.

![Code3](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/code3.png)

We create a custom dataset class to handle the histology images and their corresponding labels. This class, named ColonHistologyDataset, inherits from the Dataset class in PyTorch. We start by initializing the class with the DataFrame containing the file paths and labels, and an optional transformation parameter.

The **__len__** method returns the number of samples in the dataset. The **__getitem__** method retrieves an image and its corresponding label by index, converts the image to RGB format, and applies any specified transformations.

We define a set of transformations using **transforms.Compose** to resize the images to 224x224 pixels and convert them to tensors. These transformations are applied to the training, validation, and test datasets.

![Code4](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/code4.png)

We go ahead and design a custom **Vision Transformer (ViT)** model for image classification. This class, named **ViTForImageClassification**, extends the **nn.Module** from PyTorch. We start by initializing the model with the specified number of labels, defaulting to 8.

In the initialization method, we load a pre-trained **ViT** model from the Hugging Face library and set up a dropout layer with a 50% dropout rate. We also define a linear classifier layer that maps the hidden states to the number of labels.

The forward method processes the input pixel values through the **ViT** model, applies dropout to the output, and then passes it through the classifier to get the logits. If labels are provided, the method computes the **cross-entropy loss** between the logits and the labels. The method returns the logits and the loss (if computed).

![Code5](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/code5.png)

We set up the training parameters and initialize the **ViT** model for image classification. First, we define key hyperparameters, including the number of epochs, batch size, validation batch size, and learning rate.

Next, we identify the unique classes in the dataset to determine the number of labels for the model. The **ViTForImageClassification** model is then instantiated with the appropriate number of labels. We also utilize a pre-trained **ViT** feature extractor from the Hugging Face library.

We configure the optimizer using the Adam algorithm with the specified learning rate and define the loss function as cross-entropy loss. To leverage the computational power of GPUs, we check for CUDA availability and set the device accordingly, moving the model to the GPU if available.

![Code6](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/code6.png)

We set up the data loaders and lists to store metrics for training, validation, and testing the **ViT** model for image classification. Data loaders are configured for the training, validation, and test sets, each with specified batch sizes.

We initialize lists to store the losses and accuracies for the training, validation, and test sets. The training loop iterates over the specified number of epochs, processing each batch of training data by converting it to a **numpy** array, applying the feature extractor, and feeding it through the model. The optimizer updates the model parameters based on the calculated loss.

At the end of each epoch, we calculate the test loss and accuracy by evaluating the model on the test set without updating the model parameters. Similarly, we calculate the validation loss and accuracy by evaluating the model on the validation set.

The losses and accuracies are stored in their respective lists for plotting. Finally, we plot the training, test, and validation losses, as well as the test and validation accuracies, over the epochs.

![Code7](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/code7.png)
![Code8](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/code8.png)
![Code9](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/code9.png)


The output shows that at the end of the final epoch, the model achieved high test and validation accuracies of 97% and 96%, respectively.

![Code10](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/code10.png)
