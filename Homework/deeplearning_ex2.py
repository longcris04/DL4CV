import torch



def activation_func(x):
    #TODO Implement one of these following activation function: sigmoid, tanh, ReLU, leaky ReLU
    epsilon = 0.01   # Only use this variable if you choose Leaky ReLU
    
    result = torch.where(x>=0,x,x*epsilon)
    
    return result

def softmax(x):
    # TODO Implement softmax function here
    
    
    x = x - torch.max(x) # Minus the values in x by its max value to avoid overflow
    #print(x)
    result = torch.exp(x)
    #print(result)
    result = result/torch.sum(result)
    
    return result







def main():
    # a = torch.tensor([[2],[-1],[1]])
    # print(softmax(a))
    # print(activation_func(a))

    #torch.manual_seed(2023)

    # Define the size of each layer in the network
    num_input = 784  # Number of node in input layer (28x28)
    num_hidden_1 = 128  # Number of nodes in hidden layer 1
    num_hidden_2 = 256  # Number of nodes in hidden layer 2
    num_hidden_3 = 128  # Number of nodes in hidden layer 3
    num_classes = 10  # Number of nodes in output layer

    # Random input
    input_data = torch.randn((1, num_input),dtype=torch.double)
    #print(input_data.dtype)
    #print(f"input: {input_data}")
    # Weights for inputs to hidden layer 1
    W1 = torch.randn((num_input, num_hidden_1), dtype=torch.double)
    #print(W1.dtype)
    # Weights for hidden layer 1 to hidden layer 2
    W2 = torch.randn((num_hidden_1, num_hidden_2),dtype=torch.double)
    # Weights for hidden layer 2 to hidden layer 3
    W3 = torch.randn((num_hidden_2, num_hidden_3),dtype=torch.double)
    # Weights for hidden layer 3 to output layer
    W4 = torch.randn((num_hidden_3, num_classes),dtype=torch.double)

    
    # and bias terms for hidden and output layers
    B1 = torch.randn((1, num_hidden_1),dtype=torch.double)
    B2 = torch.randn((1, num_hidden_2),dtype=torch.double)
    B3 = torch.randn((1, num_hidden_3),dtype=torch.double)
    B4 = torch.randn((1, num_classes),dtype=torch.double)

    # TODO Calculate forward pass of the network here. Result should have the shape of [1,10]
    # Dont forget to check if sum of result = 1.0
    Z1 = input_data @ W1 + B1
    A1 = activation_func(Z1)
    Z2 = A1 @ W2 + B2
    A2 = activation_func(Z2)
    Z3 = A2 @ W3 + B3
    A3 = activation_func(Z3)
    Z4 = A3 @ W4 + B4
    
    out = softmax(Z4)
    result = out
    print(result)

if __name__ == '__main__':
    main()