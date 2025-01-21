For the colon histology report, we incorporate a comprehensive collection of histology images labeled "ColonHistology" into our Python environment. By using the Google Drive API, we mount Google Drive to access this extensive image collection. Leveraging the os library, known for its powerful file manipulation features, we enumerate the files in the specified directory, allowing us to handle the images systematically.

```python
image_directory = '/content/drive/MyDrive/ColonHistology'
     

files = os.listdir(image_directory)
files
```


Then, we generate a DataFrame from the histology image file names and their respective labels. We start by defining an empty list to store the data and a regular expression pattern to identify labels in the file names. Using the **os** library, we list all files in the specified directory and filter for those with a **.tif** extension. We then match the file names to our label pattern to extract the labels.

With the **pandas** library, we create a DataFrame containing columns for the file paths and their associated labels. To make the labels suitable for machine learning models, we utilize the **LabelEncoder** from the **sklearn** library, converting the string labels into numerical format. Finally, we construct a dictionary to map the encoded labels back to their original string representation and print the classes present in the dataset.

```python
# Create DataFrame from file names and labels
data = []
label_pattern = re.compile(r'^[a-zA-Z]+')

for filename in os.listdir(image_directory):
    if filename.endswith('.tif'):
        match = label_pattern.match(filename)
        if match:
            label = match.group(0)
            full_path = os.path.join(image_directory, filename)
            data.append([full_path, label])

df = pd.DataFrame(data, columns=['file_path', 'class'])

label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])
class_dict = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Classes in the dataset are:", class_dict)
```


We split the dataset into training, validation, and test sets to prepare it for machine learning model training and evaluation. We begin by using the **train_test_split** function from the **sklearn** library to divide the data into training/validation and test sets, with the test set comprising 15% of the total data. The split is stratified to ensure that the class distribution remains consistent across the sets.

Next, we further split the training/validation set into separate training and validation sets, with the validation set comprising approximately 5.88% of the total data. Again, we use stratified splitting to maintain the class distribution.

```python
train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df['class'], random_state=42)

train_df, val_df = train_test_split(train_val_df, test_size=0.0588, stratify=train_val_df['class'], random_state=42)

print(f'Training set size: {len(train_df)}')
print(f'Validation set size: {len(val_df)}')
print(f'Test set size: {len(test_df)}')
```

We create a custom dataset class to handle the histology images and their corresponding labels. This class, named ColonHistologyDataset, inherits from the Dataset class in PyTorch. We start by initializing the class with the DataFrame containing the file paths and labels, and an optional transformation parameter.

The **__len__** method returns the number of samples in the dataset. The **__getitem__** method retrieves an image and its corresponding label by index, converts the image to RGB format, and applies any specified transformations.

We define a set of transformations using **transforms.Compose** to resize the images to 224x224 pixels and convert them to tensors. These transformations are applied to the training, validation, and test datasets.

```python
class ColonHistologyDataset(Dataset):
  def __init__(self, dataframe, transform=None):
    self.dataframe = dataframe
    self.transform = transform

  def __len__(self):
    return len(self.dataframe)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    img_name = self.dataframe.iloc[idx, 0]
    image = Image.open(img_name).convert('RGB')
    class_label = self.dataframe.iloc[idx, 1]

    if self.transform:
      image = self.transform(image)

    return image, class_label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = ColonHistologyDataset(dataframe=train_df, transform=transform)
val_dataset = ColonHistologyDataset(dataframe=val_df, transform=transform)
test_dataset = ColonHistologyDataset(dataframe=test_df, transform=transform)
```

We go ahead and design a custom **Vision Transformer (ViT)** model for image classification. This class, named **ViTForImageClassification**, extends the **nn.Module** from PyTorch. We start by initializing the model with the specified number of labels, defaulting to 8.

In the initialization method, we load a pre-trained **ViT** model from the Hugging Face library and set up a dropout layer with a 50% dropout rate. We also define a linear classifier layer that maps the hidden states to the number of labels.

The forward method processes the input pixel values through the **ViT** model, applies dropout to the output, and then passes it through the classifier to get the logits. If labels are provided, the method computes the **cross-entropy loss** between the logits and the labels. The method returns the logits and the loss (if computed).

```python
class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=8):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
          loss_fct = nn.CrossEntropyLoss()
          loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
          return logits, loss.item()
        else:
          return logits, None
```

We set up the training parameters and initialize the **ViT** model for image classification. First, we define key hyperparameters, including the number of epochs, batch size, validation batch size, and learning rate.

Next, we identify the unique classes in the dataset to determine the number of labels for the model. The **ViTForImageClassification** model is then instantiated with the appropriate number of labels. We also utilize a pre-trained **ViT** feature extractor from the Hugging Face library.

We configure the optimizer using the Adam algorithm with the specified learning rate and define the loss function as cross-entropy loss. To leverage the computational power of GPUs, we check for CUDA availability and set the device accordingly, moving the model to the GPU if available.

```python
EPOCHS = 50
BATCH_SIZE = 10
VAL_BATCH = 1
LEARNING_RATE = 0.000001
     
unique_classes = df['class'].unique()

model = ViTForImageClassification(num_labels=len(unique_classes))

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k', do_rescale=False)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_func = nn.CrossEntropyLoss()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    model.cuda()
```

We set up the data loaders and lists to store metrics for training, validation, and testing the **ViT** model for image classification. Data loaders are configured for the training, validation, and test sets, each with specified batch sizes.

We initialize lists to store the losses and accuracies for the training, validation, and test sets. The training loop iterates over the specified number of epochs, processing each batch of training data by converting it to a **numpy** array, applying the feature extractor, and feeding it through the model. The optimizer updates the model parameters based on the calculated loss.

At the end of each epoch, we calculate the test loss and accuracy by evaluating the model on the test set without updating the model parameters. Similarly, we calculate the validation loss and accuracy by evaluating the model on the validation set.

The losses and accuracies are stored in their respective lists for plotting. Finally, we plot the training, test, and validation losses, as well as the test and validation accuracies, over the epochs.

```python
print("Number of train samples: ", len(train_df))
print("Number of test samples: ", len(test_df))
print("Detected Classes are: ", class_dict)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=VAL_BATCH, shuffle=True, num_workers=0)

# Lists to store metrics
train_losses = []
test_losses = []
val_losses = []
test_accuracies = []
val_accuracies = []

# Train the model
for epoch in range(EPOCHS):
    train_loss = 0.0
    for step, (x, y) in enumerate(train_loader):
        # Convert batch to numpy array
        x = np.array(x)
        # Apply feature extractor
        features = feature_extractor(x, return_tensors="pt")['pixel_values']
        # Send to GPU if available
        x, y = features.to(device), y.to(device)
        b_x = Variable(x)   # batch x (image)
        b_y = Variable(y)   # batch y (target)
        # Feed through model
        output, loss = model(b_x, None)
        # Calculate loss
        if loss is None:
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss += loss.item() if isinstance(loss, torch.Tensor) else loss

    # Calculate test loss and accuracy
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for test_x, test_y in test_loader:
            # Reshape and get feature matrices as needed
            test_x = feature_extractor(np.array(test_x), return_tensors="pt")['pixel_values']
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            # Generate prediction
            test_output, test_loss_batch = model(test_x, test_y)
            test_loss += test_loss_batch.item() if isinstance(test_loss_batch, torch.Tensor) else test_loss_batch
            # Predicted class value using argmax
            test_predicted_class = test_output.argmax(dim=1)
            test_correct += (test_predicted_class == test_y).sum().item()
            test_total += test_y.size(0)
    test_accuracy = test_correct / test_total
    avg_test_loss = test_loss / len(test_loader)

    # Calculate validation loss and accuracy at the end of each epoch
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_x, val_y in val_loader:
            # Reshape and get feature matrices as needed
            val_x = feature_extractor(np.array(val_x), return_tensors="pt")['pixel_values']
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            # Generate prediction
            val_output, val_loss_batch = model(val_x, val_y)
            val_loss += val_loss_batch.item() if isinstance(val_loss_batch, torch.Tensor) else val_loss_batch
            # Predicted class value using argmax
            val_predicted_class = val_output.argmax(dim=1)
            val_correct += (val_predicted_class == val_y).sum().item()
            val_total += val_y.size(0)
    val_accuracy = val_correct / val_total
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    # Store metrics for plotting
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    val_losses.append(avg_val_loss)
    test_accuracies.append(test_accuracy)
    val_accuracies.append(val_accuracy)

    print(f'Epoch: {epoch} | train loss: {avg_train_loss:.4f} | test loss: {avg_test_loss:.4f} | test accuracy: {test_accuracy:.2f} | val loss: {avg_val_loss:.4f} | val accuracy: {val_accuracy:.2f}')

# Plotting the results
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss Over Epochs')

plt.subplot(2, 1, 2)
plt.plot(epochs_range, test_accuracies, label='Test Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.legend(loc='upper left')
plt.title('Accuracy Over Epochs')

plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
```
The output shows that at the end of the final epoch, the model achieved high test and validation accuracies of 97% and 96%, respectively.

```
Epoch: 49 | train loss: 0.0486 | test loss: 0.1598 | test accuracy: 0.97 | val loss: 0.1928 | val accuracy: 0.96

```

The loss over epochs plot illustrates the training, test, and validation losses throughout the epochs. We can observe a steady decrease in all three losses, indicating that the model is effectively learning and enhancing its performance. The accuracy over epochs plot demonstrates the test and validation accuracies over the epochs. Both accuracies increase rapidly during the initial epochs and stabilize around 0.97 for the test set and 0.96 for the validation set, reflecting consistent and robust performance.

![plot](https://github.com/KadirOrcunAltunel/ColorectalCancerPrediction/blob/main/images/plot.png)
