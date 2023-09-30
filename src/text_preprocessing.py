class CharTokenizer():
    """A simple character-level tokenizer.
    
    Converts strings to integer arrays, and vice versa.
    """
    def __init__(self, all_chars: list) -> None:
        """Initializes the Encoder by generating a string-to-integer
        and integer-to-string conversion map.
        """
        self.str_to_int = {char: i for i, char in enumerate(all_chars)}
        self.int_to_str = {
            value: key for value, key in enumerate(self.str_to_int)}

    def encode(self, string: str) -> list:
        return [self.str_to_int[char] for char in string]

    def decode(self, int_arr: list) -> list:
        return ''.join([self.int_to_str[i] for i in int_arr])