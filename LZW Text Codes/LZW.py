import os
import math


class LZWCoding:
    def __init__(self, filename, data_type):
        self.filename = filename
        self.data_type = data_type
        self.codelength = None

    def compress_text_file(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        input_file = self.filename + '.txt'
        input_path = current_directory + '/' + input_file
        output_file = self.filename + '.bin'
        output_path = current_directory + '/' + output_file

        in_file = open(input_path, 'r')
        text = in_file.read().rstrip()
        in_file.close()

        encoded_text_as_integers = self.encode(text)
        encoded_text = self.int_list_to_binary_string(encoded_text_as_integers)
        encoded_text = self.add_code_length_info(encoded_text)
        padded_encoded_text = self.pad_encoded_data(encoded_text)
        byte_array = self.get_byte_array(padded_encoded_text)

        out_file = open(output_path, 'wb')
        out_file.write(bytes(byte_array))
        out_file.close()

        print(input_file + ' is compressed into ' + output_file + '.')
        uncompressed_size = len(text)
        print('Uncompressed Size: ' + '{:,d}'.format(uncompressed_size) + ' bytes')
        print('Code Length: ' + str(self.codelength))
        compressed_size = len(byte_array)
        print('Compressed Size: ' + '{:,d}'.format(compressed_size) + ' bytes')
        compression_ratio = uncompressed_size / compressed_size
        print('Compression Ratio: ' + '{:.2f}'.format(compression_ratio))

        return output_path

    def encode(self, uncompressed_data):
        dict_size = 256
        dictionary = {chr(i): i for i in range(dict_size)}
        w = ''
        result = []
        for k in uncompressed_data:
            wk = w + k
            if wk in dictionary:
                w = wk
            else:
                result.append(dictionary[w])
                dictionary[wk] = dict_size
                dict_size += 1
                w = k
        if w:
            result.append(dictionary[w])

        self.codelength = math.ceil(math.log2(len(dictionary)))
        return result

    def int_list_to_binary_string(self, int_list):
        bitstring = ''
        for num in int_list:
            for n in range(self.codelength):
                if num & (1 << (self.codelength - 1 - n)):
                    bitstring += '1'
                else:
                    bitstring += '0'
        return bitstring

    def add_code_length_info(self, bitstring):
        codelength_info = '{0:08b}'.format(self.codelength)
        bitstring = codelength_info + bitstring
        return bitstring

    def pad_encoded_data(self, encoded_data):
        if len(encoded_data) % 8 != 0:
            extra_bits = 8 - len(encoded_data) % 8
            for i in range(extra_bits):
                encoded_data += '0'
        else:
            extra_bits = 0
        padding_info = '{0:08b}'.format(extra_bits)
        encoded_data = padding_info + encoded_data
        return encoded_data

    def get_byte_array(self, padded_encoded_data):
        if (len(padded_encoded_data) % 8 != 0):
            print('The compressed data is not padded properly!')
            exit(0)
        b = bytearray()
        for i in range(0, len(padded_encoded_data), 8):
            byte = padded_encoded_data[i: i + 8]
            b.append(int(byte, 2))
        return b

    def decompress_text_file(self):
        current_directory = os.path.dirname(os.path.realpath(__file__))
        input_file = self.filename + '.bin'
        input_path = current_directory + '/' + input_file
        output_file = self.filename + '_decompressed.txt'
        output_path = current_directory + '/' + output_file

        in_file = open(input_path, 'rb')
        bytes = in_file.read()
        in_file.close()

        from io import StringIO
        bit_string = StringIO()
        for byte in bytes:
            bits = bin(byte)[2:].rjust(8, '0')
            bit_string.write(bits)
        bit_string = bit_string.getvalue()

        bit_string = self.remove_padding(bit_string)
        bit_string = self.extract_code_length_info(bit_string)
        encoded_text = self.binary_string_to_int_list(bit_string)
        decompressed_text = self.decode(encoded_text)

        out_file = open(output_path, 'w')
        out_file.write(decompressed_text)
        out_file.close()

        print(input_file + ' is decompressed into ' + output_file + '.')

        return output_path

    def remove_padding(self, padded_encoded_data):
        padding_info = padded_encoded_data[:8]
        encoded_data = padded_encoded_data[8:]
        extra_padding = int(padding_info, 2)
        if extra_padding != 0:
            encoded_data = encoded_data[:-1 * extra_padding]
        return encoded_data

    def extract_code_length_info(self, bitstring):
        codelength_info = bitstring[:8]
        self.codelength = int(codelength_info, 2)
        return bitstring[8:]

    def binary_string_to_int_list(self, bitstring):
        int_codes = []
        for bits in range(0, len(bitstring), self.codelength):
            int_code = int(bitstring[bits: bits + self.codelength], 2)
            int_codes.append(int_code)
        return int_codes

    def decode(self, encoded_values):
        dict_size = 256
        dictionary = {i: chr(i) for i in range(dict_size)}

        from io import StringIO
        result = StringIO()
        w = chr(encoded_values.pop(0))
        result.write(w)
        for k in encoded_values:
            if k in dictionary:
                entry = dictionary[k]
            elif k == dict_size:
                entry = w + w[0]
            else:
                raise ValueError('Bad compressed k: %s' % k)
            result.write(entry)
            dictionary[dict_size] = w + entry[0]
            dict_size += 1
            w = entry

        return result.getvalue()