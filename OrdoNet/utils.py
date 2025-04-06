import sys
import datetime

def log(msg):
    # Print message with timestamp
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def progress_bar(current, total, length=40):
    # Simple text progress bar
    frac = current / total
    filled = int(frac * length)
    sys.stdout.write(f"\r[{ '#' * filled}{' ' * (length - filled) }] {int(frac * 100)}%")
    sys.stdout.flush()
    if current >= total:
        print()

def plot_loss(losses):
    # Plot loss over epochs (if matplotlib is installed)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed.")
        return
    plt.plot(losses, marker='o')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()