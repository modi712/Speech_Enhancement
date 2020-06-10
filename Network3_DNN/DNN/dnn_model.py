""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F




class DNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(7175, 1025)
        self.hidden2 = nn.Linear(1025, 1025)
        self.hidden3 = nn.Linear(1025, 1025)
        
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(1025, 1025)
        
        # Define sigmoid activation and softmax output 
        self.sigmoid = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.hidden1(x)
        x = self.sigmoid(x)
        x = self.hidden2(x)
        x = self.sigmoid(x)
        x = self.hidden3(x)
        x = self.sigmoid(x)
        x = self.output(x)
        #x = self.softmax(x)
        
        return x
