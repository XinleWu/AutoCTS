from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat')

PRIMITIVES = [
    'none',
    'skip_connect',
    # 'dcc_1',
    'dcc_2',
    'trans',
    's_trans',
    'diff_gcn',
    # 'cheb_gcn',
    # 'cnn',
    # 'att1',
    # 'att2',
    # 'lstm',
    # 'gru'
]
