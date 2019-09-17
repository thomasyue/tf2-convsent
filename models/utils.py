import numpy as np

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    lst = np.arange(n, dtype="int32")
    if shuffle:
        np.random.shuffle(lst)
    iteration = 0
    start = 0
    while iteration < len(lst)//minibatch_size:
        ans = np.array(lst[start:start + minibatch_size])
        iteration += 1
        start += minibatch_size
        yield iteration-1, ans


def prepare_data_for_rnn(seqs_x, maxlen=40):
    for i in range(len(seqs_x)):
        if len(seqs_x[i]) < maxlen:
            seqs_x[i] = seqs_x[i] + ((maxlen - len(seqs_x[i])) * [0])

    return seqs_x


def prepare_data_for_cnn(seqs_x, maxlen=40, n_words=21102, filter_h=5):
    lengths_x = [len(s) for s in seqs_x]

    if maxlen != None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    pad = filter_h - 1
    x = []
    for rev in seqs_x:
        xx = []
        for i in range(pad):
            # we need pad the special <pad_zero> token.
            xx.append(n_words - 1)
        for idx in rev:
            xx.append(idx)
        while len(xx) < maxlen + 2 * pad:
            xx.append(n_words - 1)
        x.append(xx)
    x = np.array(x, dtype='int32')
    return x


def get_idx_from_sent(sent, word_idx_map):
    """
    Transforms sentence into a list of indices.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
        else:
            x.append(1)
    x.append(0)
    return x