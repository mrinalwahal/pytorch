#   1. Design model (input, output, forward pass)
#   2. Define loss function and optimizer
#   3. Train model
#       - Forward pass: compute prediction and loss
#       - Backward pass: compute gradients
#       - Update weights
#   4. Plot predictions

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Prepare datasets
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
input_size = n_features
output_size = 1

#   Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

#   Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

#   Reshape as column vector
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

#   Model
class LogisticRegression(nn.Module):
    
        def __init__(self, input_size, output_size):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, output_size)
    
        def forward(self, x):
            return torch.sigmoid(self.linear(x))

model = LogisticRegression(input_size, output_size)

#   Loss function = Binary Cross Entropy
loss = nn.BCELoss()

#   Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#   Training loop
for epoch in range(200):
    
        y_predicted = model(X_train)
        l = loss(y_predicted, y_train)
    
        #   calculate gradients
        l.backward()
    
        #   update weights
        optimizer.step()
    
        #   zero gradients
        optimizer.zero_grad()
    
        #   Only print every 10th step
        if (epoch + 1) % 10 == 0:
            print(f'{epoch+1:02d}: loss = {l.item():.6f}')

#   Plot predictions
with torch.no_grad():

    predicted = model(X_test)
    predicted_class = y_predicted.round()
    accuracy = predicted_class.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {accuracy:.2f}')

""" plt.scatter(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
plt.show() """