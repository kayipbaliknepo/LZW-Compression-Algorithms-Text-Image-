import os  # the os module is used for file and directory operations
from LZW import LZWCoding

# read and decompress the file sample.bin
filename = 'sample'
lzw = LZWCoding(filename, 'text')
output_path = lzw.decompress_text_file()

current_directory = os.path.dirname(os.path.realpath(__file__))


original_file = filename + '.txt'
original_path = current_directory + '/' + original_file


decompressed_file = filename + '_decompressed.txt'
decompressed_path = current_directory + '/' + decompressed_file


with open(original_path, 'r') as file1, open(decompressed_path, 'r') as file2:
   original_text = file1.read().rstrip()
   decompressed_text = file2.read()


if original_text == decompressed_text:
   print(original_file + ' and ' + decompressed_file + ' are the same.')
else:
   print(original_file + ' and ' + decompressed_file + ' are NOT the same.')
