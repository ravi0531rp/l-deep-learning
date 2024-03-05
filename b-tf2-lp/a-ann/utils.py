import os
import matplotlib.pyplot as plt

def plot_metrics(history, output_dir):
    """Plot training and validation metrics and save them in the output directory."""
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot training and validation metrics
    for metric in history.history.keys():
        plt.figure()
        plt.plot(history.history[metric])
        plt.plot(history.history[f'val_{metric}'])
        plt.title(f'{metric.capitalize()}')
        plt.xlabel('Epoch')
        plt.ylabel(f'{metric.capitalize()}')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        plt.close()


