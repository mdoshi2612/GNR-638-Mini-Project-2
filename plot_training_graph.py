import matplotlib.pyplot as plt

def read_loss_values(file_name):
    """
    Read loss values from a file.

    Parameters:
    - file_name: The path to the file containing loss values, one per line.

    Returns:
    A list of loss values as floats.
    """
    with open(file_name, 'r') as file:
        loss_values = [float(line.strip()) for line in file]
    return loss_values

def plot_losses(train_losses, val_losses, save_path=None):
    """
    Plot training and validation losses.

    Parameters:
    - train_losses: List of training loss values.
    - val_losses: List of validation loss values.
    - save_path: Path to save the figure (optional).
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

# Replace 'path/to/train_loss.txt' and 'path/to/val_loss.txt' with the actual paths to your files
train_loss_file = 'train_loss.txt'
val_loss_file = 'val_loss.txt'
save_path = 'losses.png'  # Specify where to save the figure

train_losses = read_loss_values(train_loss_file)
val_losses = read_loss_values(val_loss_file)

plot_losses(train_losses, val_losses, save_path)
