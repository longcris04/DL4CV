import torch
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

"""
    Create the following tensors:
        1. 3D tensor of shape 20x30x40 with all values = 0
        2. 1D tensor containing the even numbers between 10 and 100
"""

ts_zeros = torch.zeros(20,30,40)
#print(tensor, tensor.shape)
ts_even = torch.arange(10,20,2)
# print(ts_even, ts_even.shape)


"""
    x = torch.rand(4, 6)
    Calculate:
        1. Sum of all elements of x
        2. Sum of the columns of x  (result is a 6-element tensor)
        3. Sum of the rows of x   (result is a 4-element tensor)
"""
torch.manual_seed(2024)
x = torch.rand(4,6)
print(f"x: {x}")
print(f"sum all elements of x: {torch.sum(x)}")
print(f"Sum of columns of x: {torch.sum(x,dim=0)}")
print(f"Sum of rows of x: {torch.sum(x,dim=1)}")
"""
    Calculate cosine similarity between 2 1D tensor:
    x = torch.tensor([0.1, 0.3, 2.3, 0.45])
    y = torch.tensor([0.13, 0.23, 2.33, 0.45])
"""
x = torch.tensor([0.1, 0.3, 2.3, 0.45])
y = torch.tensor([0.13, 0.23, 2.33, 0.45])
csm = cosine_similarity(x,y,dim=0)
print(f"Cosine similarity: {csm}")


"""
    Calculate cosine similarity between 2 2D tensor:
    x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                      [-2.2268, 1.9799, 1.5682, 0.5850],
                      [ 1.2289, 0.5043, -0.1625, 1.1403]])
    y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                      [-0.6679, 0.0793, -2.5842, -1.5123],
                      [ 1.1110, -0.1212, 0.0324, 1.1277]])
"""
x = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                  [-2.2268, 1.9799, 1.5682, 0.5850],
                  [ 1.2289, 0.5043, -0.1625, 1.1403]])
y = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                  [-0.6679, 0.0793, -2.5842, -1.5123],
                  [ 1.1110, -0.1212, 0.0324, 1.1277]])
csm = cosine_similarity(x,y,dim=1)
print(f"Cosine similarity: {csm}")

"""
    x = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
    Make x become 1D tensor
    Then, make that 1D tensor become 3x4 2D tensor 
"""
x = torch.tensor([[ 0,  1],
                    [ 2,  3],
                    [ 4,  5],
                    [ 6,  7],
                    [ 8,  9],
                    [10, 11]])
x = x.reshape(-1)
print(f"Reshape x into 1D tensor: {x}, {x.shape}")
x = x.reshape(3,4)
print(f"Make x become 3x4 2D tensor: {x}, {x.shape}")

"""
    x = torch.rand(3, 1080, 1920)
    y = torch.rand(3, 720, 1280)
    Do the following tasks:
        1. Make x become 1x3x1080x1920 4D tensor
        2. Make y become 1x3x720x1280 4D tensor
        3. Resize y to make it have the same size as x
        4. Join them to become 2x3x1080x1920 tensor
"""
x = torch.rand(3, 1080, 1920)
y = torch.rand(3, 720, 1280)

# add first dimension to x and y
x = x.unsqueeze(0)
y = y.unsqueeze(0)
print(x.shape,y.shape)

# resize y
y = F.interpolate(y, size=(1080,1920),mode='bilinear', align_corners=False)
print(x.shape,y.shape)

# join them
stack = torch.cat((x,y),dim=0)
print(stack.shape)