import tensorflow as tf

__author__ = ['Li Tang']
__copyright__ = 'Li Tang'
__credits__ = ['Li Tang']
__license__ = 'MIT'
__version__ = '0.2.0'
__maintainer__ = ['Li Tang']
__email__ = 'litang1025@gmail.com'
__status__ = 'Production'


class SuiInitializersError(ValueError):
    pass


def get_init(initializer_name: str):
    if initializer_name == 'glorotnormal':
        return tf.initializers.GlorotNormal
    elif initializer_name == 'glorotuniform':
        return tf.initializers.GlorotUniform()
    else:
        raise SuiInitializersError
