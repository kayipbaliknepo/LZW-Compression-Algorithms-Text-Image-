# LZW Compression Algorithms (Text & Image)

This project implements the **LZW (Lempel-Ziv-Welch)** compression algorithm for both **text files** and **images**, providing lossless data compression and decompression functionality using Python.

## Features

- Compress and decompress plain text files
- Compress and decompress grayscale images (e.g., PNG, BMP)
- Educational and readable Python implementation
- Command-line usage for easy testing

## Requirements

- Python 3.11 or higher
- Pillow (`pip install pillow`) for image support

## How to Use

### Clone the Repository

```
git clone https://github.com/kayipbaliknepo/LZW-Compression-Algorithms-Text-Image-.git
cd LZW-Compression-Algorithms-Text-Image-
```

### Install Dependencies

```
pip install pillow
```

### Compress a Text File

```
python lzw_text.py compress input.txt output.lzw
```

### Decompress a Text File

```
python lzw_text.py decompress output.lzw restored.txt
```

### Compress an Image File

```
python lzw_image.py compress input.png output.lzw
```

### Decompress an Image File

```
python lzw_image.py decompress output.lzw restored.png
```

## File Structure

```
LZW-Compression/
├── lzw_text.py         # Text file compression
├── lzw_image.py        # Image file compression
├── README.md           # Project documentation
└── LICENSE             # MIT License
```

## About LZW

LZW is a lossless data compression algorithm that builds a dictionary of input patterns and encodes repeated sequences into shorter representations. It is widely used in formats like GIF and TIFF.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Author: [kayipbaliknepo](https://github.com/kayipbaliknepo)
