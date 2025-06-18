import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np  # Added: numpy for image processing

# Make sure to import your updated version of Level 5's mod 256 difference functions.
from LZW_Image_Compression import*

########################################
# Main window (GUI)
########################################
root = tk.Tk()
root.title("LZW Image Compression")
root.geometry("800x650")


########################################
# Image loading function
########################################
def load_image():
    file_path = filedialog.askopenfilename(filetypes=[("BMP Files", "*.bmp"), ("All Files", "*.*")])

    if file_path:
        try:
            # Resize for display purpose in GUI
            img = Image.open(file_path).resize((250, 250))
            img = ImageTk.PhotoImage(img)

            # Place the image on the left label
            original_label.config(image=img, text="")
            original_label.image = img  # Keep the reference to avoid GC cleanup

            # Store the selected file path in root
            root.file_path = file_path

            # Reposition the label (Tkinter sometimes needs a refresh)
            original_label.grid_forget()
            original_label.grid(row=0, column=0, padx=10)

            root.update()
            print(f"Image successfully loaded: {file_path}")

        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Error", "An error occurred while loading the image.")


########################################
# Compression button function
########################################
def compress():
    if not hasattr(root, 'file_path'):
        messagebox.showerror("Error", "Please load an image first!")
        return

    method = method_var.get()
    if not method:
        messagebox.showerror("Error", "Please select a method!")
        return

    print(f"Selected method: {method}")

    # Get original size
    original_size = os.path.getsize(root.file_path)
    compressed_file = None
    entropy = 0
    avg_code_length = 0

    # Level 2: Measurement on original grayscale
    if "Level 2" in method:
        compressed_file = compress_image_level2(root.file_path)
        # Calculate entropy and average code length based on original grayscale image
        entropy = calculate_entropy_gray(root.file_path)
        avg_code_length = calculate_avg_code_length_gray(compressed_file)

    # Level 3: Compression on grayscale difference image and measurement after decompression
    elif "Level 3" in method:
        compressed_file = compress_image_level3(root.file_path)
        restored = decompress_image_level3(compressed_file)  # Single channel (grayscale)
        entropy = calculate_entropy_level3(restored)
        avg_code_length = calculate_avg_code_length_level3(compressed_file)

    # Level 4: Color channels (256 dictionary), after decompression
    elif "Level 4" in method:
        compressed_file = compress_color_image_level4(root.file_path)
        restored = decompress_color_image_level4(compressed_file)
        entropy = calculate_entropy_color(restored)
        avg_code_length = calculate_avg_code_length_color(compressed_file)

    # Level 5: Color difference (512 dictionary, mod 256)
    elif "Level 5" in method:
        compressed_file = compress_color_image_level5(root.file_path)
        restored = decompress_color_image_level5(compressed_file)
        entropy = calculate_entropy_color_diff(restored)
        avg_code_length = calculate_avg_code_length_color_diff(compressed_file)

    else:
        messagebox.showerror("Error", "Unknown method selected!")
        return

    # If the compressed file is created, display the statistics
    if compressed_file and os.path.exists(compressed_file):
        compressed_size = os.path.getsize(compressed_file)
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        update_stats(entropy, avg_code_length, original_size, compressed_size, compression_ratio)

    messagebox.showinfo("Success", f"Compression completed with {method}!")


########################################
# Function to update statistics on GUI
########################################
def update_stats(entropy, avg_code_length, original_size, compressed_size, compression_ratio):
    stats_label.config(text=(
        f"Entropy: {entropy:.4f} bits\n"
        f"Average Code Length: {avg_code_length:.2f} bits/symbol\n"
        f"Original Size: {original_size} bytes\n"
        f"Compressed Size: {compressed_size} bytes\n"
        f"Compression Ratio: {compression_ratio:.2f}"
    ))
    stats_label.pack()
    root.update()


########################################
# Decompression button function
########################################
def decompress():
    method = method_var.get()
    if not method:
        messagebox.showerror("Error", "Please select a method!")
        return

    print(f"Selected method: {method}")

    try:
        img_path = None

        # Call the corresponding decompress function based on the selected level
        if "Level 2" in method:
            img_path = "restored_image_level2.bmp"
            decompress_image_level2("compressed_image.pkl")

        elif "Level 3" in method:
            img_path = "restored_image_level3.bmp"
            decompress_image_level3("compressed_diff_image.pkl")

        elif "Level 4" in method:
            img_path = "restored_color_image_level4.bmp"
            decompress_color_image_level4("compressed_color_image.pkl")

        elif "Level 5" in method:
            img_path = "restored_color_diff_image_level5.bmp"
            decompress_color_image_level5("compressed_color_diff_image.pkl")

        else:
            messagebox.showerror("Error", "Unknown method selected!")
            return

        # If decompressed .bmp file exists, display it on the right side
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).resize((250, 250))
            img = ImageTk.PhotoImage(img)
            decompressed_label.config(image=img, text="")
            decompressed_label.image = img
            decompressed_label.grid_forget()
            decompressed_label.grid(row=0, column=1, padx=10)
            root.update()
            print(f"Opened image: {img_path}")
            messagebox.showinfo("Success", "Image successfully opened!")
        else:
            messagebox.showerror("Error", "Output image not found!")
            print("Error: Output file not found.")

    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Error", "An error occurred while opening the image.")


########################################
# Additional Image Processing Functions
########################################
def display_in_grayscale():
    if not hasattr(root, 'file_path'):
        messagebox.showerror("Error", "Please load an image first!")
        return
    try:
        img_rgb = Image.open(root.file_path)
        img_grayscale = img_rgb.convert('L')
        img = ImageTk.PhotoImage(img_grayscale.resize((250, 250)))
        original_label.config(image=img)
        original_label.image = img
    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Error", "An error occurred while processing the image.")


def display_color_channel(channel):
    if not hasattr(root, 'file_path'):
        messagebox.showerror("Error", "Please load an image first!")
        return
    try:
        if channel == 'red':
            channel_index = 0
        elif channel == 'green':
            channel_index = 1
        elif channel == 'blue':
            channel_index = 2
        else:
            messagebox.showerror("Error", "Invalid color channel!")
            return

        img_rgb = Image.open(root.file_path)
        image_array = np.array(img_rgb)

        if len(image_array.shape) < 3:
            messagebox.showerror("Error", "Not a color image!")
            return

        # Keep only the selected channel, set others to 0
        image_array_modified = image_array.copy()
        for i in range(3):
            if i != channel_index:
                image_array_modified[:, :, i] = 0

        pil_img = Image.fromarray(image_array_modified)
        img = ImageTk.PhotoImage(pil_img.resize((250, 250)))
        original_label.config(image=img)
        original_label.image = img
    except Exception as e:
        print(f"Error: {e}")
        messagebox.showerror("Error", "An error occurred while processing the image.")


def pil_to_np(img):
    return np.array(img)


def np_to_pil(img_array):
    return Image.fromarray(np.uint8(img_array))


########################################
# UI Elements
########################################
stats_label = tk.Label(root, text="", font=("Arial", 12))
stats_label.pack()

method_options = [
    "Level 2 - Grayscale (256 dictionary)",
    "Level 3 - Grayscale + Difference Transform (512 dictionary)",
    "Level 4 - Color (256 dictionary)",
    "Level 5 - Color + Difference Transform (512 dictionary)"
]
method_var = tk.StringVar(value=method_options[0])

tk.Label(root, text="LZW Image Compression", font=("Arial", 16, "bold")).pack()

tk.Button(root, text="Load Image", command=load_image).pack()

method_frame = tk.Frame(root)
method_frame.pack()
tk.Label(method_frame, text="Compression Method: ").grid(row=0, column=0)
method_dropdown = tk.OptionMenu(method_frame, method_var, *method_options)
method_dropdown.grid(row=0, column=1)

tk.Button(root, text="Compress", command=compress).pack()
tk.Button(root, text="Open", command=decompress).pack()

# Image Areas (Left: original, Right: decompressed)
frame = tk.Frame(root)
frame.pack()
original_label = tk.Label(frame, text="Original Image", bg="gray")
original_label.grid(row=0, column=0, padx=10)
decompressed_label = tk.Label(frame, text="Decompressed Image", bg="gray")
decompressed_label.grid(row=0, column=1, padx=10)

# Additional Image Processing Buttons (Grayscale, Red, Green, Blue)
ops_frame = tk.Frame(root)
ops_frame.pack(pady=10)
tk.Button(ops_frame, text="Grayscale", width=10, command=display_in_grayscale).grid(row=0, column=0, padx=5)
tk.Button(ops_frame, text="Red", width=10, command=lambda: display_color_channel('red')).grid(row=0, column=1, padx=5)
tk.Button(ops_frame, text="Green", width=10, command=lambda: display_color_channel('green')).grid(row=0, column=2, padx=5)
tk.Button(ops_frame, text="Blue", width=10, command=lambda: display_color_channel('blue')).grid(row=0, column=3, padx=5)

root.mainloop()
