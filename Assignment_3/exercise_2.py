# TODO: Add any imports you deem necessary

def get_encoding_old(alphabet: dict, base: int = 2) -> dict:
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
            code_dict[a] = toBase(counter - 1)[2:]  # sliced the '0b' prefix off the bin() numbers
            return code_dict
        while has_prefix(toBase(counter), encoding_set):
            counter += 1
        code_dict[a] = toBase(counter)[2:]  # sliced the '0b' prefix off the bin() numbers
        encoding_set.add(toBase(counter))
        counter += base
    return code_dict  # added final return case to fix a linter error


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
    alphabet_list = list(alphabet.items())
    alphabet_list = sorted(alphabet_list, key=lambda x: x[1], reverse=True)
    code_dict = dict()
    tree = split_list_shannon(alphabet_list)
    for alpha, code in flatten(tree):
        code_dict[alpha] = code
    return code_dict


def split_list(lst, parts):
    length = len(lst)
    return [lst[i * length // parts: (i + 1) * length // parts] for i in range(parts)]


def split_list_shannon(lst, code=""):
    print(lst)
    if len(lst) == 1:
        return lst[0][0], code
    lst_branch = []
    while sum(x[1] for x in lst_branch) + lst[0][1] <= sum(x[1] for x in lst) - lst[0][1]:
        lst_branch.append(lst.pop(0))
        if len(lst) == 0:
            print("lst == 0")
            head = lst_branch.pop(0)
            return [split_list_shannon([head], code=code + "0"), split_list_shannon(lst, code=code + "1")]
    return [split_list_shannon(lst_branch, code=code + "0"),
            split_list_shannon(lst, code=code + "1")]


def flatten(lst):
    """flatten trees/nested lists"""
    for item in lst:
        if isinstance(item, list):
            for nested_item in flatten(item):
                yield nested_item
        else:
            yield item


def has_prefix(string, strings):
    for s in strings:
        if string.startswith(s):
            return True
    return False


def toBase(x, base=2):
    return bin(x)
