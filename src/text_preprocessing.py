class Encoder():
    def __init__(self, all_chars):
        self.str_to_int = {char: i for i, char in enumerate(all_chars)}
        self.int_to_str = {
            value: key for value, key in enumerate(self.str_to_int)}

    def encode(self, string):
        return [self.str_to_int[char] for char in string]

    def decode(self, int_arr):
        return ''.join([self.int_to_str[i] for i in int_arr])