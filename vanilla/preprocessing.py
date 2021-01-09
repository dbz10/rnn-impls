import numpy as np


def create_vocabulary(input_text):
    flat = ' '.join(input_text)
    vocabulary = list(set(flat))
    vocab_size = len(vocabulary)
    char_to_idx = {char: idx for idx, char in enumerate(vocabulary)}
    idx_to_char = {idx: char for idx, char in enumerate(vocabulary)}
    return {'vocab_size': vocab_size, 'encoder': char_to_idx, 'decoder': idx_to_char}


def train_test_split(lines, test_size=0.1):
    """
    Split data into a train and test set. Set test size to control the fraction
    of rows in the test set.
    :param Xs:
    :param ys:
    :param test_size:
    :return:
    """
    assert (0 < test_size < 1), "test_size must be between 0 and 1 (exclusive)"
    n_examples = len(lines)
    test_samples = int(round(n_examples * test_size))
    train_samples = n_examples - test_samples

    idxs = np.arange(len(lines))
    # mutable shuffling of indices
    np.random.shuffle(idxs)

    train_idxs = idxs[:train_samples]
    test_idxs = idxs[train_samples:]

    l_train = lines[train_idxs]
    l_test = lines[test_idxs]

    return l_train, l_test


def text_to_input_and_target(text_lines):
    flattened = ' '.join(text_lines)
    xs = flattened[:-1]
    ys = flattened[1:]
    return xs, ys
