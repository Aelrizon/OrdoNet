import sys
import datetime

def log(msg):
    """
    Prints a message with a timestamp.
    If msg is not a string, it's converted automatically.
    """
    try:
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
    except Exception as e:
        print("Error in log:", e)

def progress_bar(current, total, length=40):
    """
    Displays a simple text progress bar.
    
    current: current progress value.
    total: total value (must be > 0).
    length: total length of the bar in characters.
    """
    try:
        if total == 0:
            sys.stdout.write("\rTotal is zero, cannot compute progress.")
            sys.stdout.flush()
            return
        frac = current / total
        # Ensure fraction is within [0, 1]
        frac = max(0, min(frac, 1))
        filled = int(frac * length)
        bar = f"\r[{'#' * filled}{' ' * (length - filled)}] {int(frac * 100)}%"
        sys.stdout.write(bar)
        sys.stdout.flush()
        if current >= total:
            print()  # Move to next line when complete.
    except Exception as e:
        print("\nError in progress_bar:", e)

def plot_loss(losses):
    """
    Plots the loss over epochs using matplotlib.
    If losses is empty or matplotlib is not installed, prints a message.
    """
    if not losses:
        print("No loss values provided to plot.")
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed.")
        return
    try:
        plt.plot(losses, marker='o')
        plt.title("Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Error while plotting loss:", e)