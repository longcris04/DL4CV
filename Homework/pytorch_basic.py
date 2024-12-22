import torch
import torch.nn as nn
import numpy as np
import math
'''
1. Create a torch.tensor

-empty Tensor:
>> torch.empty(3,3)  # 3x3 uninitialized tensor

-Tensor with specific value:
>> torch.tensor([[1, 2], [3, 4]])  # 2x2 tensor with specified values

-Zeros or Ones:
>> tensor_zeros = torch.zeros(3, 3)  # 3x3 tensor of zeros
>> tensor_ones = torch.ones(3, 3)    # 3x3 tensor of ones

-Random Values:
>> tensor_random = torch.rand(3, 3)  # 3x3 tensor with random values in [0, 1)
>> tensor_normal = torch.randn(3, 3) # 3x3 tensor with values from normal distribution

-Tensor with a particular shape filled with a scalar value:
>> tensor_full = torch.full((3, 3), 7)  # 3x3 tensor filled with 7s

-Identity Matrix:
>> tensor_eye = torch.eye(3)  # 3x3 identity matrix

-Using .new_* Methods:
>> tensor_existing = torch.tensor([[1, 2], [3, 4]])
>> tensor_zeros_like = tensor_existing.new_zeros(2, 2)  # 2x2 tensor of zeros with the same type as `tensor_existing`

2. Check shape and dtype

-Check shape:
>> print(tensor.shape)  # Output: torch.Size([2, 3])

-Check data type:
>> print(tensor.dtype)  # Output: torch.int64 (or other dtype based on your tensor)

3. Indexing and Slicing (same as numpy array)

>> tensor = torch.tensor([[10, 20, 30], [40, 50, 60]])

-Basic indexing:
>> tensor[0,1] # Access the element at row 0, column 1. Output: 20

4. Reshaping (same as numpy array)  

-Add dimension at specified position:
>>

5. Tranpose or Permute

-Rearrange dimension
>> tensor = torch.randn(2, 3, 4)  # Shape: (2, 3, 4)
>> transposed_tensor = tensor.permute(1, 2, 0) # Move the first dimension (0) to the last, resulting in (3, 4, 2)

-Swap dimension
>> transposed_tensor = tensor.transpose(0, 2) # Swap dimension 0 with dimension 2, resulting in shape (4, 3, 2)

6. Modify dimension

-Add dimension:
>> tensor = torch.tensor([1, 2, 3])  # Shape: (3,)
>> tensor.unsqueeze(0)  # Shape: (1, 3), Add a dimension at position 0 (making it a 2D tensor)
>> tensor.unsqueeze(1)  # Shape: (3, 1), Add a dimension at position 1

-Remove dimension:
>> tensor.squeeze(0) # Remove shape at position 0

7. Concatenation

>> tensor1 = torch.tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
>> tensor2 = torch.tensor([[5, 6], [7, 8]])  # Shape: (2, 2)

-Concatenate along dimension 0 (rows):
>> concat_dim0 = torch.cat((tensor1, tensor2), dim=0)  # Shape: (4, 2)

-Concatenate along dimension 1 (columns):
>> concat_dim1 = torch.cat((tensor1, tensor2), dim=1)  # Shape: (2, 4)

-Stack tensors along a new dimension (e.g., dim=0):
>> stacked = torch.stack((tensor1, tensor2), dim=0)  # Shape: (2, 2, 2)

8. Max Min

>> a = torch.randn(4,3)
>> max_val = torch.amax(a, dim=(0, 1), keepdim=True)

9. Type conversion

>> x_int = x.to(torch.int32) # Convert to a different data type (e.g., float32 to int)

10. Matrix multiplication and Broadcasting

-Matrix multiplication:
>> A @ B

-Broadcasting:
>> A * 1 # Same as numpy but the broadcasted tensor only expand dimension to outer

10. Random

>> torch.manual_seed(42)


'''

torch.manual_seed(42)
# a = torch.randn(2,3,5)
# print(a)
# max_val = torch.amin(a, dim=0, keepdim=False)
# #max_val = max_val.reshape(-1,1)
# print(max_val)
# print(max_val.shape)

# A = torch.tensor([[[[1,2],[1,2]],[[1,2],[1,2]],[[1,2],[1,2]]]])
# B = torch.tensor([[1,0,2],[0,1,2]])
# print(A.shape,B.shape)

# C1 = A @ B
# #C2 = torch.mm(A,B)
# C3 = torch.matmul(A,B)
# print(f"C1: {C1}")
# print(C1.shape)
# #print(f"C2: {C2}")
# print(f"C3: {C3}")
# print(C3.shape)

# A = torch.tensor([[[1,2,3],[1,2,3]]])
# print(A,A.dtype,A.shape)
# B = A* torch.tensor([[[2],[3]]])

# print(f"B: {B}",B.shape)

A = torch.tensor([[0],[-1],[1]])

print(A - torch.max(A))
# s = torch.sum(A)
# print(A.shape)
