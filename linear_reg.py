import torch
import torch.nn as nn
# x is height in inch and y is weight
x=torch.tensor([140,155,159,179,192,200,212],dtype=torch.float32)
y=torch.tensor([60,62,67,70,71,72,75],dtype=torch.float32)

print(f"Shape of x: {x.shape}")
print(f"Shape of y: {y.shape}")
print("")

x=x/200.0
x_reshape=x.unsqueeze(1)
print(f"x_reshape:{x_reshape}")
y_reshape=y.unsqueeze(1)
print(f"y_reshape:{y}")

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression,self).__init__()
        self.Linear=nn.Linear(1,1)

    def forward(self,x):
        return self.Linear(x)
    
model=LinearRegression()
loss_fn=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

epochs=1000
for epoch in range(epochs):
    y_pred=model(x_reshape)

    loss=loss_fn(y_pred,y_reshape)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch:{epoch}, Loss:{loss.item():.4f}")   


weight=model.Linear.weight.item()
bias=model.Linear.bias.item()

print("model trained")
print(f"Weight: {weight:.4f}")
print(f"Bias:{bias:.4f}")

test_height=torch.tensor([[212.0/200.0]])
predicted_weight=model(test_height)
print(f"Predicted height for 160: {predicted_weight.item():.2f}")