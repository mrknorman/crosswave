from tensorflow.keras.layers import (Layer, Input, Conv1D, 
                                    ReLU, MaxPooling1D, GlobalAveragePooling1D, 
                                    Conv1DTranspose, Concatenate,
                                    Dense, Dropout,
                                    MultiHeadAttention, LayerNormalization,
                                    UpSampling1D
                                    )
from tensorflow.keras.models import Model
import tensorflow as tf

class PositionalEncoding(Layer):
    def __init__(self, max_len=4096, d_model=128, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model
        self.pe = self.create_positional_encoding()

    def create_positional_encoding(self):
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / self.d_model))

        sin_values = tf.sin(position * div_term)
        cos_values = tf.cos(position * div_term)

        sin_values_expanded = tf.expand_dims(sin_values, 1)
        cos_values_expanded = tf.expand_dims(cos_values, 1)

        pe = tf.concat([sin_values_expanded, cos_values_expanded], axis=1)
        pe = tf.reshape(pe, (1, self.max_len, self.d_model))

        return pe

    def call(self, x):
        return x + self.pe[:, :tf.shape(x)[1], :]

# Blocks
class Encoder(Layer):
    def __init__(self, filters = None, kernel_sizes = None, pool_sizes = None, strides = None, **kwargs):

        if filters is None:
            filters = [64, 32, 32, 16, 16, 16]
        if kernel_sizes is None:
            kernel_sizes = [8, 8, 8, 8, 8, 8]
        if pool_sizes is None:
            pool_sizes = [None, 8, None, None, 8, None, None]
        if strides is None:
            strides = [None, 8, None, None, 8, None, None]

        super(Encoder, self).__init__(**kwargs)
        self.conv_layers = []
        self.pool_layers = []

        for i, (f, k) in enumerate(zip(filters, kernel_sizes)):
            self.conv_layers.append(Conv1D(filters=f, kernel_size=k, padding='same', activation='relu'))

            if i < len(pool_sizes) and i < len(strides) and (pool_sizes[i] is not None):
                self.pool_layers.append(MaxPooling1D(pool_size=pool_sizes[i], strides=strides[i]))

    def call(self, inputs):
        x = inputs
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x)
            if i < len(self.pool_layers):
                x = self.pool_layers[i](x)
        return x

class Decoder(Layer):
    def __init__(self, filters, kernel_sizes, upsample_sizes, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.conv_layers = []
        self.upsample_layers = []

        for i, (f, k) in enumerate(zip(reversed(filters), reversed(kernel_sizes))):
            if i < len(upsample_sizes) and (upsample_sizes[i] is not None):
                self.upsample_layers.append(UpSampling1D(size=upsample_sizes[i]))
            self.conv_layers.append(Conv1D(filters=f, kernel_size=k, padding='same', activation='relu'))

    def call(self, inputs):
        x = inputs
        upsample_index = 0
        for i, conv in enumerate(self.conv_layers):
            if upsample_index < len(self.upsample_layers) and i == len(self.conv_layers) - len(self.upsample_layers) + upsample_index:
                x = self.upsample_layers[upsample_index](x)
                upsample_index += 1
            x = conv(x)
        return x

def build_feature_extractor():
    return tf.keras.Sequential([Encoder()])

def build_denoising_autoencoder():
    filters = [64, 32, 32, 16, 16, 16]
    kernel_sizes = [8, 8, 8, 8, 8, 8]
    pool_sizes = [None, 8, None, None, 8, None, None]
    strides = [None, 8, None, None, 8, None, None]

    inputs = Input(shape=(16384, 1))

    # Encoder
    encoded = Encoder(filters, kernel_sizes, pool_sizes, strides)(inputs)
    decoded = Decoder(filters, kernel_sizes, pool_sizes)(encoded)

    autoencoder = Model(inputs=inputs, outputs=decoded)
    return autoencoder

class AttentionBlock(Layer):
    def __init__(self, head_size, num_heads, num_dense_neurons, dropout=0, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.num_dense_neurons = num_dense_neurons
        self.dropout = dropout

        # Layers
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)
        self.multi_head_attention = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)
        self.dense1 = Dense(num_dense_neurons, activation="relu")

    @tf.function
    def call(self, inputs):

        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and len(inputs) == 2:
            input_1, input_2 = inputs
        else:
            input_1 = inputs
            input_2 = inputs

        # Normalization and Attention
        x = self.layernorm1(input_1)
        y = self.layernorm2(input_2)

        x = self.multi_head_attention(x, y)
        x = Dropout(self.dropout)(x)
        res = x + input_1

        # Feed Forward Part
        x = self.layernorm3(res)
        x = self.dense1(x)
        x = Dropout(self.dropout)(x)
        x = self.dense2 (x)
        x = Dropout(self.dropout)(x)

        return x + res

    def build(self, input_shape):
        # Set the size of the second dense layer dynamically based on the input shape
        if (isinstance(input_shape, tuple) or isinstance(input_shape, list)) and len(input_shape) == 2:
            output_size = input_shape[0][-1]
        else:
            output_size = input_shape[-1]

        self.dense2 = Dense(output_size, activation = "relu")
        super(AttentionBlock, self).build(input_shape)