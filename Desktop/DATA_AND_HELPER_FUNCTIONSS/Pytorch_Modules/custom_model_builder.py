import torch.nn as nn

class CustomCNN(nn.Module):

  def __init__(self, input_shape, hidden_units, output_shape):
    '''# -modify the model structure in the backend in whatever way you want to use it.
       # -number of blocks
       # -number of layers inside each block
       # -kernel size, stride and padding of each layer
       # -whether relu is required or not
       # -whether maxpool is required or not
       # -kernel size inside maxpool
       # -size of the linear layer in the classifier'''
    super().__init__()

    self.cnn_block1 = nn.Sequential(

        nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1), #layer 1 in block 1
        nn.ReLU(),

        nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1), #layer 2 in block 1
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2)
    )

    self.cnn_block2 = nn.Sequential(

        nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1), #layer 1 in block 2
        nn.ReLU(),

        nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1), #layer 2 in block 2
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2)
    )

    '''self.cnn_block3 = nn.Sequential(

        nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1), #layer 1 in block 3
        nn.ReLU(),

        nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1), #layer 2 in block 3
        nn.ReLU(),

        nn.MaxPool2d(kernel_size=2)
    )'''

    self.classifier = nn.Sequential(

        nn.Flatten(), #flatten the previous layer size
        nn.Linear(hidden_units * 16 * 16, output_shape) #putting in feedforward linear layer
    )

  def forward(self, x): #forward all the layers
      out = self.cnn_block1(x)
      out = self.cnn_block2(out)
      #out = self.cnn_block3(out)
      out = self.classifier(out)
      return out
