from dataset import load_csv, normalize
from network import Network
from utils import log, progress_bar, plot_loss
from loss import Loss
from optimizer import AdamOptimizer

# Set path to the dataset (raw string for Windows)
data_file = r"C:\IT\VSCODE\Simple Project\xxx\Salary_dataset.csv"
log(f"Loading data from {data_file}")

# Load data from CSV
# CSV format: [Index, YearsExperience, Salary]
raw_data, raw_targets = load_csv(data_file, delimiter=",", has_header=True)

# Use YearsExperience as input (column index 1)
data = [[row[1]] for row in raw_data]
# Use Salary as target (convert to list)
targets = [[t] for t in raw_targets]

log("Normalizing data...")
# Scale inputs and targets to [0, 1]
data = normalize(data)
targets = normalize(targets)

# Build the network: 1 input → 4 hidden → 1 output
net = Network([1, 4, 1])
log("Network created.")

# Set up Adam optimizer
adam = AdamOptimizer(size=net.total_parameters(), lr=0.01)

epochs = 100
loss_history = []

log("Training started with Adam optimizer...")
for epoch in range(epochs):
    total_loss = 0
    for i, (inp, targ) in enumerate(zip(data, targets)):
        # Forward pass
        output = net.forward(inp)
        # Calculate MSE loss
        loss = Loss.mse(targ, output)
        total_loss += loss
        # Backpropagation — get gradients
        grads = net.backward(targ)
        # Get current weights and biases
        params = net.get_parameters()
        # Apply Adam update
        new_params = adam.update(params, grads)
        # Load updated params back into the network
        net.set_parameters(new_params)
        progress_bar(i + 1, len(data))
    avg_loss = total_loss / len(data)
    loss_history.append(avg_loss)
    log(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

# Try prediction for 5 years of experience
test_input = [5.0]
# Normalize input just like training data
norm_test_input = normalize([[test_input[0]]])[0]
prediction = net.predict(norm_test_input)
log(f"Prediction for input {test_input}: {prediction}")

# Show loss chart (if matplotlib is installed)
plot_loss(loss_history)

# Save model to file
net.save("salary_model.txt")
log("Model saved to salary_model.txt")