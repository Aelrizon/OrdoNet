from network import Network
from loss import Loss
from optimizer import AdamOptimizer
from utils import log, progress_bar, plot_loss
from dataset import normalize
import math

# Generate sine wave data
num_samples = 100
# Generate x values evenly spaced from 0 to 2π
x_values = [i * (2 * math.pi / (num_samples - 1)) for i in range(num_samples)]
# Compute y = sin(x)
y_values = [math.sin(x) for x in x_values]

# Format data: each input and target is a single-element list
data = [[x] for x in x_values]
targets = [[y] for y in y_values]

log("Normalizing sine wave data...")
# Normalize inputs and targets to [0, 1]
data = normalize(data)
targets = normalize(targets)

# Build the network: 1 input, 6 hidden neurons, 1 output
net = Network([1, 6, 1])
log("Sine network created.")

# Initialize Adam optimizer
adam = AdamOptimizer(size=net.total_parameters(), lr=0.01)

epochs = 300
loss_history = []

log("Training sine network with Adam optimizer...")
for epoch in range(epochs):
    total_loss = 0
    for i, (inp, targ) in enumerate(zip(data, targets)):
        # Forward pass
        output = net.forward(inp)
        # Compute Mean Squared Error
        loss = Loss.mse(targ, output)
        total_loss += loss
        # Backpropagation: compute gradients
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
    log(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

# Test prediction for a new value, e.g. x = π/4
test_input = [math.pi / 4]
# Normalize test input like training data
norm_test_input = normalize([[test_input[0]]])[0]
prediction = net.predict(norm_test_input)
log(f"Prediction for input {test_input}: {prediction}")

# Plot loss over epochs (if matplotlib is installed)
plot_loss(loss_history)

# Save the trained model
net.save("sine_model.txt")
log("Sine model saved to sine_model.txt")