# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a Convolutional Neural Network (CNN) for image classification using the Fashion-MNIST dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve high accuracy while maintaining efficiency.

## Neural Network Model

<img width="1271" height="463" alt="exp3 model" src="https://github.com/user-attachments/assets/949340fa-f108-4704-8ce3-2b31cf993db1" />

## DESIGN STEPS

STEP 1: Problem Statement

Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

STEP 2:Dataset Collection

Use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits.

STEP 3: Data Preprocessing

Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

STEP 4:Model Architecture

Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

STEP 5:Model Training

Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

STEP 6:Model Evaluation

Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

STEP 7: Model Deployment & Visualization

Save the trained model, visualize predictions, and integrate it into an application if needed.


## PROGRAM

### Name: KIRUBA R C
### Register Number: 212224230125
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128*3*3,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,10)

  def forward(self,x):
    x = self.pool(torch.relu(self.conv1(x)))
    x = self.pool(torch.relu(self.conv2(x)))
    x = self.pool(torch.relu(self.conv3(x)))
    x = x.view(x.size(0),-1)
    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = self.fc3(x)
    return x


```

```python
# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):

     for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images,labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    print('Name: THIRUNAVUKKARASU MEENAKSHISUNDARAM')
    print('Register Number: 212224220117')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

        
        
        
        print('Name: THIRUNAVUKKARASU MEENAKSHISUNDARAM')
        print('Register Number:212224220117')
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

## OUTPUT
### Training Loss per Epoch

<img width="371" height="279" alt="image" src="https://github.com/user-attachments/assets/4b3e2211-2e35-4070-8000-447baf589e59" />


### Confusion Matrix

<img width="960" height="813" alt="image" src="https://github.com/user-attachments/assets/d6c54289-191d-4be0-8f49-27d4f16d661b" />


### Classification Report

<img width="611" height="387" alt="image" src="https://github.com/user-attachments/assets/d2d01ec2-2565-4aef-bf07-fc59e853bc0c" />


### New Sample Data Prediction

<img width="602" height="654" alt="image" src="https://github.com/user-attachments/assets/c9224417-6795-4221-aa35-cfc4a5d4d9c4" />


## RESULT
Thus, We have developed a convolutional deep neural network for image classification to verify the response for new images.
