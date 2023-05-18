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
    alphabet_list = list(alphabet.items())
    alphabet_list = sorted(alphabet_list, key=lambda x: x[1], reverse=True)
    code_dict = dict()
    tree = shannon(alphabet_list)
    for alpha, code in flatten(tree):
        code_dict[alpha] = code
    return code_dict


def split_list(lst):
    lst_branch = []
    while sum(x[1] for x in lst_branch) < sum(x[1] for x in lst):  # and
        branch_sum = sum(x[1] for x in lst_branch)
        lst_sum = sum(x[1] for x in lst)
        lst_branch.append(lst.pop(0))
        branch_sum_new = sum(x[1] for x in lst_branch)
        lst_sum_new = sum(x[1] for x in lst)
        if abs(branch_sum_new - lst_sum_new) > abs(branch_sum - lst_sum):
            # revert lists
            lst = [lst_branch.pop(-1)] + lst
            break
    return lst_branch, lst


def shannon(lst, code=""):
    if len(lst) == 1:
        return lst[0][0], code
    split_lst = split_list(lst)
    return [shannon(split_lst[0], code=code + "0"),
            shannon(split_lst[1], code=code + "1")]


def flatten(lst):
    """flatten trees/nested lists"""
    for item in lst:
        if isinstance(item, list):
            for nested_item in flatten(item):
                yield nested_item
        else:
            yield item
