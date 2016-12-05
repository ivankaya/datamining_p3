import numpy as np

def mapper(key, value):
    print "Mapping..."

    # Read data
    dat = np.zeros([len(value), 250], dtype='f')
    count = 0
    for i in value:
        dat[count, :] = i
        count = count + 1

    # key: None
    # value: one line of input file
    yield "key", "value"  # this is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    yield np.random.randn(200, 250)
