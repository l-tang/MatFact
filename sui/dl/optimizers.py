import tensorflow as tf


class SuiOptimizersError(ValueError):
    pass


def get_opti(opti_name: str):
    if opti_name == 'adam':
        return tf.keras.optimizers.Adam
    else:
        raise SuiOptimizersError
