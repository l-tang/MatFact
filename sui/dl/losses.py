import tensorflow as tf


class SuiLossesError(ValueError):
    pass


def get_loss(loss_name: str):
    if loss_name == 'sigmoid':
        return tf.keras.losses.BinaryCrossentropy
    elif loss_name == 'mse':
        return tf.keras.losses.MSE
    else:
        raise SuiLossesError
