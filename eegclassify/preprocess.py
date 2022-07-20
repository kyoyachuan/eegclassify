import numpy as np


def jitter(x, sigma=0.03):
    # https://arxiv.org/pdf/1706.00527.pdf
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    # https://arxiv.org/pdf/1706.00527.pdf
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[1]))
    return np.multiply(x, factor[:, :, np.newaxis])


def random_flip(x):
    flip = np.random.choice([-1, 1], size=(x.shape[0]))
    return np.multiply(flip, x)


def random_window_slice(x, reduce_ratio=0.9):
    # https://halshs.archives-ouvertes.fr/halshs-01357973/document
    target_len = np.ceil(reduce_ratio*x.shape[-1]).astype(int)
    if target_len >= x.shape[1]:
        return x
    starts = np.random.randint(low=0, high=x.shape[-1]-target_len, size=(x.shape[0])).astype(int)
    ends = (target_len + starts).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[1]):
            ret[i, dim, :] = np.interp(np.linspace(0, target_len, num=x.shape[-1]),
                                       np.arange(target_len), pat[dim, starts[i]:ends[i]]).T
    return ret


def random_shift(x):
    shift = np.random.randint(low=-x.shape[-1]/8, high=x.shape[-1]/8)
    x = np.roll(x, shift, axis=-1)
    return x
