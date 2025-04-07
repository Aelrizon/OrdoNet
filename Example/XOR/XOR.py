from network import Network
from loss import Loss
from optimizer import AdamOptimizer
from utils import log, progress_bar, plot_loss
from dataset import normalize  # We'll use normalize even if XOR values are 0/1

# Define XOR dataset manually
data = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
# XOR target: output is 0 when inputs are the same, 1 otherwise
targets = [
    [0],
    [1],
    [1],
    [0]
]

log("Normalizing XOR data...")
# Although values are 0 and 1, we call normalize to follow the same interface.
data = normalize(data)
targets = normalize(targets)

# Build the network: 2 inputs, 2 hidden neurons, 1 output
net = Network([2, 2, 1])
log("XOR Network created.")

# Initialize Adam optimizer for the network
adam = AdamOptimizer(size=net.total_parameters(), lr=0.05)

epochs = 500
loss_history = []

log("Training XOR network with Adam optimizer...")
for epoch in range(epochs):
    total_loss = 0
    for i, (inp, targ) in enumerate(zip(data, targets)):
        output = net.forward(inp)
        loss = Loss.mse(targ, output)
        total_loss += loss
        # Backpropagation: get gradients
        grads = net.backward(targ)
        # Get current parameters (flattened)
        params = net.get_parameters()
        # Update parameters using Adam
        new_params = adam.update(params, grads)
        # Set updated parameters back to the network
        net.set_parameters(new_params)
        progress_bar(i + 1, len(data))
    avg_loss = total_loss / len(data)
    loss_history.append(avg_loss)
    if (epoch + 1) % 50 == 0:
        log(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

# Test predictions on XOR inputs
for inp, targ in zip(data, targets):
    prediction = net.predict(inp)
    log(f"XOR Input: {inp}, Prediction: {prediction}, Target: {targ}")

# Plot loss over epochs (if matplotlib is installed)
plot_loss(loss_history)

# Save the XOR model to file
net.save("xor_model.txt")
log("XOR model saved to xor_model.txt")