# TODO: Add any imports you deem necessary

def get_encoding(alphabet: dict, base: int = 2) -> dict:
    """ Generate an prefix encoding for each element of the alphabet

    Args:
    alphabet - a dictionary that holds in its keys the alphabet item and
                   its frequency as the value, e.g. {'a': 0.5, 'b': 0.5}
    base - an `int` that represents the base of the code. Feel free to make
           your implementation open to other bases. For the purposes of this
           exercise, using only base `2`, i.e. binary code, suffices.

    Returns a `dict` with alphabet items in its keys and the encoding as the
    values, e.g. {'a': '0', 'b': '1'}
    """
    counter = 0
    code_dict = dict()
    encoding_set = set()
    alphabet_list = list(alphabet.items())
    alphabet_list = sorted(alphabet_list, key=lambda x: x[1])
    for i, (a, p) in enumerate(alphabet_list):
        if i == len(alphabet_list) - 1:
            code_dict[a] = toBase(counter - 1)[2:] # sliced the '0b' prefix off the bin() numbers
            return code_dict
        while has_prefix(toBase(counter), encoding_set):
            counter += 1
        code_dict[a] = toBase(counter)[2:] # sliced the '0b' prefix off the bin() numbers
        encoding_set.add(toBase(counter))
        counter += base
    return code_dict # added final return case to fix a linter error

def has_prefix(string, strings):
    for s in strings:
        if string.startswith(s):
            return True
    return False


def toBase(x, base=2):
    return bin(x)
