import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import matplotlib.ticker as ticker
from pathlib import Path


def show_histogram(image_path, save_folder=None, suffix=""):
    """Display and save image histogram as a vector graphic with publication-quality styling."""
    # Open image file
    image = Image.open(image_path)

    # Convert image to grayscale
    gray_image = image.convert('L')

    # Get pixel values
    pixels = list(gray_image.getdata())

    # Set seaborn style for a clean look without grid
    sns.set_style("white")
    plt.rcParams['font.family'] = 'Arial'  # Use Arial for a professional look
    plt.rcParams['font.size'] = 12  # Set font size for readability
    plt.rcParams['axes.linewidth'] = 1.2  # Slightly thicker axes for clarity

    # Clear current figure to prevent overlap
    plt.clf()

    # Create figure with specific size suitable for publications
    plt.figure(figsize=(6, 4), dpi=300)

    # Plot histogram with refined styling
    plt.hist(pixels, bins=256, range=(0, 255), color='#4C78A8', alpha=0.85, edgecolor='none')

    # Customize axes
    plt.xlabel('Gray Value', fontsize=14, labelpad=10)
    plt.ylabel('Frequency (×10³)', fontsize=14, labelpad=10)

    # Format y-axis to show frequency in thousands
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x * 1e-3:.1f}'))

    # Set axis limits
    ax.set_xlim(0, 255)

    # Add major and minor ticks on all four axes
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=4, top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in', length=2, top=True, bottom=True, left=True, right=True)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))  # Minor ticks on x-axis
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))  # Minor ticks on y-axis

    # Ensure labels are only on bottom and left to avoid clutter
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_label_position('bottom')
    ax.yaxis.set_label_position('left')

    # Adjust layout to prevent clipping
    plt.tight_layout()

    # Save as high-quality PDF if save_folder is specified
    if save_folder:
        file_name = os.path.basename(image_path)
        name, ext = os.path.splitext(file_name)
        save_path = os.path.join(save_folder, f"{name}_{suffix}.pdf")
        plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)

    # Close figure to free memory
    plt.close()


# Define image paths
root_path = Path(__file__).resolve().parent.parent
plain_path = str(root_path / "Result" / "InputImage" / "Cameraman.bmp")
cipher_path = str(root_path / "Result" / "OutputImage" / "DcEncryption" / "QF=90" / "Cameraman.bmp" / "36.jpg")

# Common save folder path
save_folder = r"./"

# Generate and save histograms without titles or grid, with ticks on all four sides
show_histogram(plain_path, save_folder=save_folder, suffix="plain")
show_histogram(cipher_path, save_folder=save_folder, suffix="cipher")
