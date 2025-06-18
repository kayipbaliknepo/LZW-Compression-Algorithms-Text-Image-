from PIL import Image
import numpy as np
import os
import pickle
import math
from collections import Counter

# ----------------------------------------------------
# 1) COMMON HELPER FUNCTIONS (DICTIONARY BASED COMPRESSION)
# ----------------------------------------------------
def compress_256(uncompressed):
    dict_size = 256
    dictionary = {(i,): i for i in range(dict_size)}
    w = []
    result = []
    for c in uncompressed:
        wc = w + [c]
        if tuple(wc) in dictionary:
            w = wc
        else:
            result.append(dictionary[tuple(w)])
            dictionary[tuple(wc)] = dict_size
            dict_size += 1
            w = [c]
    if w:
        result.append(dictionary[tuple(w)])
    return result

def decompress_256(compressed):
    dict_size = 256
    dictionary = {i: [i] for i in range(dict_size)}
    w = dictionary[compressed.pop(0)]
    result = w.copy()
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + [w[0]]
        else:
            raise ValueError(f'Bad compressed k: {k}')
        result.extend(entry)
        dictionary[dict_size] = w + [entry[0]]
        dict_size += 1
        w = entry
    return result

def compress_512(uncompressed):
    dict_size = 512
    dictionary = {(i,): i for i in range(dict_size)}
    w = []
    result = []
    for c in uncompressed:
        wc = w + [c]
        if tuple(wc) in dictionary:
            w = wc
        else:
            result.append(dictionary[tuple(w)])
            dictionary[tuple(wc)] = dict_size
            dict_size += 1
            w = [c]
    if w:
        result.append(dictionary[tuple(w)])
    return result

def decompress_512(compressed):
    dict_size = 512
    dictionary = {i: [i] for i in range(dict_size)}
    w = dictionary[compressed.pop(0)]
    result = w.copy()
    for k in compressed:
        if k in dictionary:
            entry = dictionary[k]
        elif k == dict_size:
            entry = w + [w[0]]
        else:
            raise ValueError(f'Bad compressed k: {k}')
        result.extend(entry)
        dictionary[dict_size] = w + [entry[0]]
        dict_size += 1
        w = entry
    return result

# ----------------------------------------------------
# 2) COMMON FILE SIZE AND IMAGE COMPARISON FUNCTIONS
# ----------------------------------------------------
def calculate_metrics(original_path, compressed_file):
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_file)
    ratio = original_size / compressed_size
    print(f"Original file size: {original_size} bytes")
    print(f"Compressed file size: {compressed_size} bytes")
    print(f"Compression ratio: {ratio:.2f}")

def compare_images_gray(img1_path, img2_path):
    arr1 = np.array(Image.open(img1_path).convert('L'))
    arr2 = np.array(Image.open(img2_path).convert('L'))
    if np.array_equal(arr1, arr2):
        print("Grayscale: Original and decompressed image are exactly the same.")
    else:
        diff = np.abs(arr1 - arr2)
        print(f"Grayscale: Images differ. Average difference: {np.mean(diff)}")

def compare_images_color(img1_path, img2_path):
    arr1 = np.array(Image.open(img1_path).convert('RGB'))
    arr2 = np.array(Image.open(img2_path).convert('RGB'))
    if np.array_equal(arr1, arr2):
        print("Color: Original and decompressed image are exactly the same.")
    else:
        diff = np.abs(arr1.astype(np.int16) - arr2.astype(np.int16))
        print(f"Color: Images differ. Average difference: {np.mean(diff)}")

# ----------------------------------------------------
# 3) LEVEL 2: GRAYSCALE (ORIGINAL VALUES)
# ----------------------------------------------------
def compress_image_level2(image_path):
    img = Image.open(image_path).convert('L')
    data = list(img.getdata())
    compressed = compress_256(data)
    compressed_data = {
        'gray': compressed,
        'size': img.size
    }
    filename = 'compressed_image.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(compressed_data, f)
    print("Level 2: Compression completed and saved to '{}' file.".format(filename))
    return filename

def decompress_image_level2(compressed_file):
    with open(compressed_file, 'rb') as f:
        compressed_data = pickle.load(f)
    gray = decompress_256(compressed_data['gray'])
    size = compressed_data['size']
    img = Image.new('L', size)
    img.putdata(gray)
    output = 'restored_image_level2.bmp'
    img.save(output)
    print("Level 2: Image successfully decompressed and saved as '{}'.".format(output))
    return img

def calculate_entropy_gray(image_path):
    img = Image.open(image_path).convert('L')
    data = list(img.getdata())
    total = len(data)
    freq = Counter(data)
    entropy = -sum((count / total) * math.log2(count / total) for count in freq.values())
    print(f"Level 2: Entropy: {entropy:.4f} bit")
    return entropy

def calculate_avg_code_length_gray(compressed_file):
    with open(compressed_file, 'rb') as f:
        data = pickle.load(f)
    compressed_size = len(data['gray']) * 8  # assuming 8-bit
    original_size = data['size'][0] * data['size'][1]
    avg = compressed_size / original_size
    print(f"Level 2: Average Code Length: {avg:.2f} bit/symbol")
    return avg

# ----------------------------------------------------
# 4) LEVEL 3: GRAYSCALE + DIFFERENCE TRANSFORM (512 DICTIONARY)
# ----------------------------------------------------
def get_difference_image_level3(image_array):
    """
    For each pixel:
      - (0,0): original value
      - (i,0) (first column, i>0): difference from the pixel in the previous row, same column (mod 256)
      - Other pixel (i,j) (j>0): difference from the pixel to the left (mod 256)
    """
    height, width = image_array.shape
    diff_image = np.zeros_like(image_array, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                diff_image[i, j] = image_array[i, j]
            elif j == 0:
                diff_image[i, j] = (int(image_array[i, j]) - int(image_array[i-1, j])) % 256
            else:
                diff_image[i, j] = (int(image_array[i, j]) - int(image_array[i, j-1])) % 256
    return diff_image

def restore_image_from_difference_level3(diff_image):
    """
    Restoring the original image from the difference image:
      - (0,0): direct diff value
      - (i,0) (first column, i>0): add diff to the pixel above and mod 256
      - Other pixel (i,j) (j>0): add diff to the restored pixel to the left and mod 256
    """
    height, width = diff_image.shape
    restored = np.zeros_like(diff_image, dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if i == 0 and j == 0:
                restored[i, j] = diff_image[i, j]
            elif j == 0:
                restored[i, j] = (int(diff_image[i, j]) + int(restored[i-1, j])) % 256
            else:
                restored[i, j] = (int(diff_image[i, j]) + int(restored[i, j-1])) % 256
    return restored

def compress_image_level3(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    # Apply new difference transform:
    diff_img = get_difference_image_level3(img_array)
    diff_flat = diff_img.flatten().tolist()
    # LZW compression with 512 dictionary:
    compressed = compress_512(diff_flat)
    compressed_data = {
        'diff': compressed,
        'size': img_array.shape
    }
    filename = 'compressed_diff_image.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(compressed_data, f)
    print("Level 3: Compression completed and saved to '{}' file.".format(filename))
    return filename

def decompress_image_level3(compressed_file):
    with open(compressed_file, 'rb') as f:
        data = pickle.load(f)
    diff_flat = decompress_512(data['diff'])
    diff_img = np.array(diff_flat, dtype=np.uint8).reshape(data['size'])
    # Restore original image from difference image:
    restored = restore_image_from_difference_level3(diff_img)
    output = 'restored_image_level3.bmp'
    Image.fromarray(restored).save(output)
    print("Level 3: Image successfully decompressed and saved as '{}'.".format(output))
    return restored

def calculate_entropy_level3(image_array):
    pixel_values = image_array.flatten()
    total = len(pixel_values)
    unique, counts = np.unique(pixel_values, return_counts=True)
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))
    print(f"Level 3: Entropy: {entropy:.4f} bit")
    return entropy

def calculate_avg_code_length_level3(compressed_file):
    with open(compressed_file, 'rb') as f:
        data = pickle.load(f)
    compressed_size = len(data['diff']) * 9  # assuming 9-bit
    original_size = data['size'][0] * data['size'][1]
    avg = compressed_size / original_size
    print(f"Level 3: Average Code Length: {avg:.2f} bit/symbol")
    return avg

# ----------------------------------------------------
# 5) LEVEL 4: COLOR (ORIGINAL RGB VALUES, 256 DICTIONARY)
# ----------------------------------------------------
def compress_color_image_level4(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)
    r = img_array[:, :, 0].flatten().tolist()
    g = img_array[:, :, 1].flatten().tolist()
    b = img_array[:, :, 2].flatten().tolist()
    compressed_r = compress_256(r)
    compressed_g = compress_256(g)
    compressed_b = compress_256(b)
    compressed_data = {
        'r': compressed_r,
        'g': compressed_g,
        'b': compressed_b,
        'size': img_array.shape[:2]
    }
    filename = 'compressed_color_image.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(compressed_data, f)
    print("Level 4: Compression completed and saved to '{}' file.".format(filename))
    return filename

def decompress_color_image_level4(compressed_file):
    with open(compressed_file, 'rb') as f:
        data = pickle.load(f)
    r = np.array(decompress_256(data['r'])).reshape(data['size'])
    g = np.array(decompress_256(data['g'])).reshape(data['size'])
    b = np.array(decompress_256(data['b'])).reshape(data['size'])
    img = np.stack((r, g, b), axis=2).astype(np.uint8)
    output = 'restored_color_image_level4.bmp'
    Image.fromarray(img, 'RGB').save(output)
    print("Level 4: Image successfully decompressed and saved as '{}'.".format(output))
    return img

def calculate_entropy_color(image_array):
    pixels = image_array.flatten()
    total = len(pixels)
    unique, counts = np.unique(pixels, return_counts=True)
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))
    print(f"Level 4: Entropy: {entropy:.4f} bit")
    return entropy

def calculate_avg_code_length_color(compressed_file):
    with open(compressed_file, 'rb') as f:
        data = pickle.load(f)
    compressed_size = (len(data['r']) + len(data['g']) + len(data['b'])) * 8
    original_size = data['size'][0] * data['size'][1] * 3
    avg = compressed_size / original_size
    print(f"Level 4: Average Code Length: {avg:.2f} bit/symbol")
    return avg

# ----------------------------------------------------
# 6) LEVEL 5: COLOR + DIFFERENCE TRANSFORM
# ----------------------------------------------------
def get_difference_image_level5(channel):
    """
    - (r=0, c=0) pixel: takes the original value directly.
    - (r=0, c>0) pixel: difference from the left pixel (mod 256).
    - (r>0, c=0) pixel: difference from the above pixel (mod 256).
    - In other cases, difference from the left pixel (mod 256).
    """
    rows, cols = channel.shape
    diff_image = np.zeros_like(channel, dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                # Write the top left pixel directly
                diff_image[r, c] = channel[r, c]
            elif r == 0 and c > 0:
                # First row: difference from the left pixel
                val = (int(channel[r, c]) - int(channel[r, c - 1]) + 256) % 256
                diff_image[r, c] = val
            elif c == 0 and r > 0:
                # First column: difference from the above pixel
                val = (int(channel[r, c]) - int(channel[r - 1, c]) + 256) % 256
                diff_image[r, c] = val
            else:
                # Remaining pixels: difference from the left pixel
                val = (int(channel[r, c]) - int(channel[r, c - 1]) + 256) % 256
                diff_image[r, c] = val

    return diff_image

def restore_image_from_difference_level5(diff_channel):
    """
    Restore the original image from the mod 256 difference image:
      - First pixel: direct diff value
      - First row (c>0): left pixel + diff
      - First column (r>0): above pixel + diff
      - Others: left pixel + diff
    """
    rows, cols = diff_channel.shape
    restored = np.zeros_like(diff_channel, dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                restored[r, c] = diff_channel[r, c]
            elif r == 0 and c > 0:
                restored[r, c] = (restored[r, c - 1] + diff_channel[r, c]) % 256
            elif c == 0 and r > 0:
                restored[r, c] = (restored[r - 1, c] + diff_channel[r, c]) % 256
            else:
                restored[r, c] = (restored[r, c - 1] + diff_channel[r, c]) % 256

    return restored

def compress_color_image_level5(image_path):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # Apply difference to each color channel as in Level 3, but with mod 256
    diff_r = get_difference_image_level5(img_array[:, :, 0])
    diff_g = get_difference_image_level5(img_array[:, :, 1])
    diff_b = get_difference_image_level5(img_array[:, :, 2])

    # Compress with 512 dictionary
    compressed_r = compress_512(diff_r.flatten().tolist())
    compressed_g = compress_512(diff_g.flatten().tolist())
    compressed_b = compress_512(diff_b.flatten().tolist())

    compressed_data = {
        'r': compressed_r,
        'g': compressed_g,
        'b': compressed_b,
        'size': img_array.shape[:2]
    }
    filename = 'compressed_color_diff_image.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(compressed_data, f)
    print("Level 5: Compression completed and saved to '{}' file.".format(filename))
    return filename

def decompress_color_image_level5(compressed_file):
    with open(compressed_file, 'rb') as f:
        data = pickle.load(f)

    # Decompress the difference images
    diff_r = np.array(decompress_512(data['r'])).reshape(data['size'])
    diff_g = np.array(decompress_512(data['g'])).reshape(data['size'])
    diff_b = np.array(decompress_512(data['b'])).reshape(data['size'])

    # Restore original channels from differences
    restored_r = restore_image_from_difference_level5(diff_r)
    restored_g = restore_image_from_difference_level5(diff_g)
    restored_b = restore_image_from_difference_level5(diff_b)

    # Combine channels and save
    img = np.stack((restored_r, restored_g, restored_b), axis=2).astype(np.uint8)
    output = 'restored_color_diff_image_level5.bmp'
    Image.fromarray(img, 'RGB').save(output)
    print("Level 5: Image successfully decompressed and saved as '{}'.".format(output))
    return img

def calculate_entropy_color_diff(image_array):
    pixels = image_array.flatten()
    total = len(pixels)
    freq = np.bincount(pixels, minlength=512)
    probs = freq[freq > 0] / total
    entropy = -np.sum(probs * np.log2(probs))
    print(f"Level 5: Entropy: {entropy:.4f} bit")
    return entropy

def calculate_avg_code_length_color_diff(compressed_file):
    with open(compressed_file, 'rb') as f:
        data = pickle.load(f)
    compressed_size = (len(data['r']) + len(data['g']) + len(data['b'])) * 9
    original_size = data['size'][0] * data['size'][1] * 3
    avg = compressed_size / original_size
    print(f"Level 5: Average Code Length: {avg:.2f} bit/symbol")
    return avg

# ----------------------------------------------------
# Main Functions (Level by Level)
# ----------------------------------------------------
def main_level2():
    image_path = 'big_image_grayscale.bmp'
    compressed_file = compress_image_level2(image_path)
    calculate_metrics(image_path, compressed_file)
    calculate_entropy_gray(image_path)
    calculate_avg_code_length_gray(compressed_file)
    decompress_image_level2(compressed_file)
    compare_images_gray(image_path, 'restored_image_level2.bmp')

def main_level3():
    image_path = 'big_image_grayscale.bmp'
    compressed_file = compress_image_level3(image_path)
    calculate_metrics(image_path, compressed_file)
    restored = decompress_image_level3(compressed_file)
    calculate_entropy_level3(restored)
    calculate_avg_code_length_level3(compressed_file)
    compare_images_gray(image_path, 'restored_image_level3.bmp')

def main_level4():
    image_path = 'big_image_grayscale.bmp'
    compressed_file = compress_color_image_level4(image_path)
    calculate_metrics(image_path, compressed_file)
    restored = decompress_color_image_level4(compressed_file)
    calculate_entropy_color(restored)
    calculate_avg_code_length_color(compressed_file)
    compare_images_color(image_path, 'restored_color_image_level4.bmp')

def main_level5():
    image_path = 'big_image.bmp'
    compressed_file = compress_color_image_level5(image_path)
    calculate_metrics(image_path, compressed_file)
    restored_image = decompress_color_image_level5(compressed_file)
    calculate_entropy_color_diff(restored_image)
    calculate_avg_code_length_color_diff(compressed_file)
    compare_images_color(image_path, 'restored_color_diff_image_level5.bmp')


main_level2()