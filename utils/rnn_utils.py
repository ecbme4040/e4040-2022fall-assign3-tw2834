import tensorflow as tf

def one_hot(input_val, vocab_len):
    return tf.one_hot(tf.cast(input_val, tf.int32), vocab_len, on_value=1, off_value=0, axis=-1)
