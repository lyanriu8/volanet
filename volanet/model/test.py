import torch

# --------------------------------------------
#               INTRODUCTION
# --------------------------------------------

# ---- tensor creation ----
# matrix is usally defined as row x col (height x width)
#1: creation directly
data = [[1,2,3], [4,5,6]]
my_tensor = torch.tensor(data)
# output:
# tensor([[1, 2, 3],
#         [4, 5, 6]])

#2: creation from shape
shape = (2,3)
ones = torch.ones(shape) #tensor of ones
# output: 
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])

#3: creation from mimicking another tensor
template = torch.tensor([[1,2,3], [4,5,6]])
rand_like = torch.randn_like(template, dtype=torch.float)
# output:
# tensor([[-0.4665,  0.1745, -0.4777],
#         [-0.0204,  0.0706,  0.5964]])

# ---- tensor attributes ----
# 1. shape - tuple describing dimensions of tensor
# 2. dtype - datatype of tensor - must be float to for model to learn
# 3. device - where the tensor lives (cpu or gpu)

# ---- tensors and autograd ----
# causes pytorch to build computation graph
x_data = torch.tensor([[3.0], [4.0]], requires_grad= True) # marks tensors as parameters/bias to modify therefore must be of float type
y_data = torch.tensor([[1.0], [2.0]], requires_grad=True)
z_data = x_data + y_data
# print(z_data.grad_fn) ---> allows backtracking to what operation created it
# <AddBackward0 object at 0x122cdfe80>

# ---- * vs @ ----
# *: element wise multiplication -> two tensors that multiply tgt must have the same shape
# @: matrix-multiplication: m1 cols = m2 rows -> m1 @ m2 = matrix with rows m1 x cols m2
m1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
m2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
matrix_product = m1 @ m2
# output:
# tensor([[ 58,  64],
#         [139, 154]])

# ---- how to collapse a tensor ----
# dim=0 collapses rows -> flatter vertically (pancake)
# dim=1 collapses cols -> flattern horizontally (accordian)
scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
avg_per_assignment = scores.mean(dim=0) #avg for each assignment we collapse vertically because each col represents an assignment
# output: 
# tensor([2.5000, 3.5000, 4.5000])

avg_per_student = scores.mean(dim=1) # avg for each student means each row represents ONE student
# output: 
# tensor([2., 5.])

# ---- basic indexing ----
# Standard NumPy indexing for getting blocks of data
x = torch.arange(12).reshape(3,4) # 3 x 4 matrix
col_2 = x[:, 2] # this gets all rows from column 2
# output:
# tensor([ 2,  6, 10])

# ---- dynamic indexing ----
# find the index of highest value -> how you get a models final prediction
scores = torch.tensor([
    [10,  0, 30, 20, 1], 
    [ 1, 30,  2,  5, 0]
]) 

best_indicies = torch.argmax(scores, dim=1) # flattens horizontally to get hights value
# output: 
# tensor([3, 1])

# ---- .gather() ----
# if u need a custoom selection for indecies, this is optimized
data = torch.tensor([
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33]
])

# shopping list of which column to get from each row
indicies_to_select = torch.tensor([
    [2],
    [0],
    [3]
])

# go row by row and GATHER specified cols
selected_values = torch.gather(data, dim=1,index=indicies_to_select)

# output:
# tensor([[12],         from row 0, got index 2
#         [20],         from row 1, got index 0
#         [33]])        from row 2, got index 3


# --------------------------------------------
#            BUILDING THE MODEL
# --------------------------------------------

# ---- (1) forward pass ----
# model is initiall random so it must guess
# implementing the first guess

# model: simple linear regression
# y = XW + b
# y: prediction  X: input   W: weight   b: bias  ----->  (W and b are adjustible parameters)
# goal: get our PREDICTED y as close as possible to REAL y

# EXAMPLE: fake data that follows line y=2x+1 with some noise
# test batch will have 10 data points
N = 10

# each data point has 1 input feature and 1 output feature
D_in = 1
D_out = 1

# create input data X
X = torch.randn(N, D_in) # N rows and D_in cols

# create "true" (actual) target labels y by using "true" W and b ---> creating labels
# the "true" W is 2.0 and the "true" b is 1.0
true_W = torch.tensor([[2.0]])
true_b = torch.tensor([[1.0]])
y_true = X @ true_W + true_b + torch.randn(N, D_out) * 0.1 # this is added noise
# [10 x 1] @ [1 x 1] + [1 x 1] = [10 x 1] 

# IMPORTANT: The model will never see true_W or true_b ---> the model must LEARN true_W and true_b 
# by ONLY looking at X and y_true

W = torch.randn(D_in, D_out, requires_grad=True) # init W = tensor([[-0.4292]], requires_grad=True)
b = torch.randn(1, requires_grad=True)           # init b = tensor([0.3533], requires_grad=True)
# these are both randown tensors that represent model's initial hypothesis

y_hat = X @ W + b

print(f"Predicted y_hat: \n {y_hat[:3]} \n")
print(f"true y: \n {y_true[:3]} \n")

# ---- (2) the backward pass ----
# the forward pass was the guess, the backward pass is the post-guess analysis of what went wrong
# we compare guess to true values and we figure out which DIRECTION to tune our parameters (W and b)
# loss (cost function): function returns 1 value that quantifies the error (how off the model was)
# Mean Squared Error (MSE): pretty much standard deviation
error = y_hat - y_true
squared_error = error ** 2
loss = squared_error.mean()

print(f"Loss: {loss}")
# goal: make the loss as SMALL as possible
# loss tensor has grad_fn ---> source from which all knowledge will flow backward
# autograd: backpropagates and tunes all W and b
# we tell pytorch to travel backward from 'loss' and calculate the GRADIENT for all parameters with 'requires_grad=True' 
# specifically ---> gradient of loss wrt to W and gradient of loss wrt to b

loss.backward()
# the .grad attribute tells us how to adjust its respective parameters
print(f"gradient for W: {W.grad} \n")
print(f"gradient for b: {b.grad} \n")

# -ve gradient means we must increase parameter to decrease loss
# +ve gradient means we must decrease parameter to decrease loss

# ---- (3) the training loop ----
# gradient descent: describes how our model incrementally decreases loss
# theta_(t+1) = theta_t-n*gradient_theta_wrt_L
# theta: parameters (W and b)
# n: learning rate ---> how big of a steop we take
# gradient: gradient calculated thru .backward()
# specifically ---> W_new = W_old - learning_rate * W.grad 
#              ---> b_new = b_old - learning_rate * b.grad

# training loop: repeat 3 steps for many epochs (training cycles)

# hyper parameters
learning_rate, epochs = 0.01, 0

# re-init parameters
W, b = torch.randn(D_in, D_out, requires_grad=True), torch.randn(1, requires_grad=True) 

# training loop
for epoch in range(epochs):
    
    # forward pass and loss
    y_hat = X @ W + b
    loss = torch.mean((y_hat - y_true)**2)
    
    # backward pass
    loss.backward()
    assert W.grad is not None
    assert b.grad is not None
    
    # no_grad() is needed because we do not want pytorch to build a computation graph
    with torch.no_grad():     # 
        W -= learning_rate * W.grad 
        b -= learning_rate * b.grad
    
    # zero gradients
    # pytorch adds gradients by default, but we want a new gradient for each iteration and adjust
    assert W.grad is not None
    assert b.grad is not None
    W.grad.zero_()
    b.grad.zero_()
    
    if epoch % 10 == 0:
        print(f"epoch={epoch:02d}: loss={loss.item():.4f}, W={W.item():.3f}, b={b.item():.3f}")

print(f"final parameters: W={W.item():.3f}, b={b.item():.3f}")
print("true parameters: W=2.000, b=1.000")

# output: 
# epoch=00: loss=0.2949, W=1.266, b=1.127
# epoch=10: loss=0.2359, W=1.338, b=1.103
# epoch=20: loss=0.1896, W=1.403, b=1.084
# epoch=30: loss=0.1533, W=1.460, b=1.068
# epoch=40: loss=0.1246, W=1.512, b=1.056
# epoch=50: loss=0.1019, W=1.558, b=1.046
# epoch=60: loss=0.0839, W=1.599, b=1.039
# epoch=70: loss=0.0696, W=1.636, b=1.033
# epoch=80: loss=0.0582, W=1.669, b=1.028
# epoch=90: loss=0.0491, W=1.699, b=1.024
# final parameters: W=1.723, b=1.022
# true parameters: W=2.000, b=1.000



# --------------------------------------------
#              TORCH.NN MODULE
# --------------------------------------------
# pytoch pre-built layers that are the foundation of modern, professional models

# (1) torch.nn.Linear
# Neatly packages linear regression in a standardized object

# input has 1 feature, output has 1 value
D_in = 1
D_out = 1

# create linear layer (LEGO brick)
linear_layer = torch.nn.Linear(in_features=D_in, out_features=D_out)

# can see pre-set parameters
print(f"layer weight (W): {linear_layer.weight} \n")
print(f"layer bias   (b): {linear_layer.bias}   \n")

# use the layer like a function
# X is prev defined tensor of shape [10, 1]
y_hat_nn = linear_layer(X)
print(f"output of nn.Linear (first 3 rows): \n {y_hat_nn[:3]}")

# parameter: a special tensor that has require_grad=True, auto-registers with model, handles all
# book keeping for us

# linear layers are always going to be linear 

# ---- activation funcitons ----
# layers between linear layers that give the model non-linearity so that it can learn not just along
# the straight

# 1. ReLU ---> ReLU(x) = max(0, x)
relu = torch.nn.ReLU()
sample_data = torch.tensor([-2.0, -0.4, 0.0, 0.5, 2.0])
activated_data = relu(sample_data)
# output: 
# Original Data: tensor([-2.0000, -0.4000,  0.0000,  0.5000,  2.0000])
# Data after ReLU: tensor([0.0000, 0.0000, 0.0000, 0.5000, 2.0000])


# 2. GeLU ---> more gentle ReLU and is used for Transformers (GPT, Llama)
gelu = torch.nn.GELU()
sample_data = torch.tensor([-2.0, -0.4, 0.0, 0.5, 2.0])
activated_data = gelu(sample_data)

print(f"Original Data: {sample_data}")
print(f"Data after ReLU: {activated_data}")

# 3. softmax ----> used on final output layer for classification
# covert logits (raw model scores) into a probablity distribution so that each output is given
# a probability
softmax = torch.nn.Softmax(dim=-1)

# Raw model scores for 2 items across 4 classes
logits = torch.tensor([[1.0, 3.0, 0.5, 1.5],
                       [-1.0, 2.0, 1.0, 0.0]])

probabilities = softmax(logits)
print(f"Output Probabilities: \n {probabilities}")
print(f"Sum of probabilities for item 1: {probabilities[0].sum()}")

# 4. embedding ----> used in all LLMs to vectorize words
# 5. layernorm ----> used to stablize range to preven exploding/vanishing gradient 
#              ----> turns mean of each output vector 0 and the std to 1

# 6. dropout ----> randomly zeros out neurons to make model more robust and not reliant on 1 neuron
# ONLY USED DURING TRAINING
dropout_layer = torch.nn.Dropout(p=0.5)
input_tensor = torch.ones(1, 10)

# activate dropout layer for training -> outputs are randomly zerod and scaled
dropout_layer.train()
output_train = dropout_layer(input_tensor)
print(output_train)

# decactivates for evaluation/prediction
dropout_layer.eval()
output_eval = dropout_layer(input_tensor)
print(output_eval)

# --------------------------------------------
#               TORCH.OPTIM
# --------------------------------------------
# nn.module: defines instructions and architecture of model
# torch.optim: how to efficiently train the model

# ---- professional model ----
import torch.nn as nn

# Inherit from nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # We define the the layers we are going to use in the constructor
        self.linear_layer = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        # In the forward pass, we CONNECT the layers
        return self.linear_layer(x)

# instantiate model
model = LinearRegressionModel(in_features=1, out_features=1)
print(model)

# ---- professional trainer ----
import torch.optim as optim

# hyperparameters
learning_rate = 0.1

# create ADAM optimizer
# pass model.parameters() to tell it which tensors to manage
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# we can also use a pre-built loss function from torch.nn
loss_fn = nn.MSELoss()      # Mean Squared Error Loss

# now both we have the model and the model trainer

# ---- professional training loop ----
# three line banger
# 1. 
# optimizer.zero_grad()

# 2. 
# loss.backward()

# 3. 
# optimizer.step()

epochs = 100

for epoch in range(epochs):
    # 1. forward pass
    y_hat = model(X)
    
    # 2. calculate loss
    loss = loss_fn(y_hat, y_true)
    
    # 3. line mantra
    # 1. zero gradients 
    optimizer.zero_grad()

    # 2. compute gradients
    loss.backward()

    # 3. update parameters
    optimizer.step()
    
    # (optional) print to check progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch:02d}: Loss={loss.item():.4f}")

    

