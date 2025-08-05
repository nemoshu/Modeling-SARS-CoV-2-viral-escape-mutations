import numpy as np
import tensorflow as tf

def get_angles(pos, i, d_model):
    """
    Generates a matrix of angles for positional encoding.

    Args:
        pos (np.ndarray): position indices
        i (np.ndarray): embedding dimension indices
        d_model (int): dimensionality of positional encoding

    Returns:
        angles (np.ndarray): angles for positional encoding
    """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """
    Generates a tensor for sinusoidal positional encoding.

    Args:
        position (int): maximum sequence length
        d_model (int): embedding dimension

    Returns:
        encodings (np.ndarray): sinusoidal positional encoding, shape (1, position, d_model)
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)
    # Apply sine to even indices in the array.
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # Apply cosine to odd indices in the array.
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(q, k, v, mask):
    """
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output (tf.Tensor): attention-weighted sum over v
        attention_weights (tf.Tensor): normalized weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # Scale matmul_qk.
    dk = tf.cast(tf.shape(k)[-1], tf.float16)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        """
        Multi-head attention layer.

        Args:
            d_model (int): hidden dimension
            num_heads (int): number of attention heads

        Returns:
            None
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert((d_model % self.num_heads) == 0)

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
        (batch_size, num_heads, seq_len, depth)

        Args:
            x (tf.Tensor): input tensor
            batch_size (int): batch size for each training batch

        Returns:
            x (tf.Tensor): output tensor with dimensions (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """
        Apply multi-head attention.

        Args:
            v (tf.Tensor): value input tensor, shape (batch_size, seq_len, d_model)
            k (tf.Tensor): key input tensor, shape (batch_size, seq_len, d_model)
            q (tf.Tensor): query input tensor, shape (batch_size, seq_len, d_model)
            mask (tf.Tensor): mask tensor

        Returns:
            output (tf.Tensor): output tensor, shape (batch_size, seq_len_q, d_model)
            attention_weights (tf.Tensor): normalized weights
        """
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads,
                                             #  seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads,
                                             #  seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads,
                                             #  seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])

        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """
    Provides a point-wise feed forward layer.

    Args:
        d_model (int): input/output dimension
        dff (int): inner-layer dimension

    Returns:
        output (tf.keras.Sequential): Point-wise feed forward layer
    """
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        """
        Encoder layer.

        Args:
            d_model (int): input/output dimension
            num_heads (int): number of attention heads
            dff (int): inner layer dimension
            rate (float): dropout rate
            **kwargs: keyword arguments to be passed to super(EncoderLayer, self).__init__()

        Returns:
            None
        """
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, mask):
        """
        Applies multi-head self-attention and feed-forward attention.

        Args:
            x (tf.Tensor): input tensor, shape (batch_size, input_seq_len, d_model)
            mask (tf.Tensor): mask tensor

        Returns:
            output (tf.Tensor): output tensor, shape (batch_size, input_seq_len, d_model)
        """
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)

        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        """
        Intializes the decoder layer.

        Args:
            d_model (int): input/output dimension
            num_heads (int): number of attention heads
            dff (int): inner layer dimension
            rate (float): dropout rate
            **kwargs: keyword arguments to be passed to super(DecoderLayer, self).__init__()

        Returns:
            None
        """
        super(DecoderLayer, self).__init__(**kwargs)

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        """
        Applies self-attention, cross-attention with enc_output and feed-forward attention.

        Args:
            x (tf.Tensor): input tensor, shape (batch_size, target_seq_len, d_model)
            enc_output (tf.Tensor): output from encoder, shape (batch_size, d_model)
            look_ahead_mask (tf.Tensor): mask tensor
            padding_mask (tf.Tensor): masks padding in output

        Returns:
            output: (tf.Tensor): output tensor, shape (batch_size, target_seq_len, d_model)
            attn_weights_block_1 (tf.Tensor): self-attention weights
            attn_weights_block_2 (tf.Tensor): cross-attention weights
        """
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        # (batch_size, target_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, rate=0.1, **kwargs):
        """
        Full encoder

        Args:
            num_layers (int): number of layers
            d_model (int): input dimension
            num_heads (int): number of attention heads
            dff (int): inner layer dimension
            input_vocab_size (int): vocabulary size for embedding
            maximum_position_encoding (int): maximum sequence length
            rate (float): dropout rate
            **kwargs: keyword arguments to be passed to super(Encoder, self).__init__()

        Returns:
            None
        """
        super(Encoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)


        self.enc_layers = [ EncoderLayer(d_model, num_heads, dff, rate)
                            for _ in range(num_layers) ]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, mask):
        """
        Implements the full transformer encoder

        Args:
            x (tf.Tensor): input tensor, shape (batch_size, input_seq_len, d_model)
            mask (tf.Tensor): mask tensor

        Returns:
            output: (tf.Tensor): encoded output tensor, shape (batch_size, input_seq_len, d_model)
        """
        seq_len = tf.shape(x)[1]

        # Add embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask)

        return x # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 target_vocab_size, maximum_position_encoding,
                 rate=0.1, **kwargs):
        """
        Full decoder.

        Args:
            num_layers (int): number of layers
            d_model (int): input dimension
            num_heads (int): number of attention heads
            dff (int): inner layer dimension
            target_vocab_size (int): vocabulary size for embedding
            maximum_position_encoding (int): maximum sequence length
            rate (float): dropout rate
            **kwargs: keyword arguments to be passed to super(Decoder, self).__init__()

        Returns:
            None
        """
        super(Decoder, self).__init__(**kwargs)

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(
            target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
            maximum_position_encoding, d_model)

        self.dec_layers = [ DecoderLayer(d_model, num_heads, dff, rate)
                            for _ in range(num_layers) ]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, look_ahead_mask, padding_mask):
        """
        Implements the full transformer decoder

        Args:
            x (tf.Tensor): input tensor, shape (batch_size, target_seq_len, d_model)
            enc_output (tf.Tensor): output from encoder, shape (batch_size, input_seq_len, d_model)
            look_ahead_mask (tf.Tensor): mask tensor
            padding_mask (tf.Tensor): masks padding in output

        Returns:
            output: (tf.Tensor): output tensor, shape (batch_size, target_seq_len, d_model)
            attention_weights (tf.Tensor): attention weights
        """
        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output,
                look_ahead_mask, padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class Transformer(tf.keras.Model):
    def __init__(
            self,
            input_vocab_size,
            target_vocab_size,
            seq_len_input,
            seq_len_target,
            num_layers=6,
            d_model=512,
            num_heads=8,
            dff=2048,
            dropout_rate=0.1,
            **kwargs
    ):
        """
        Transformer model.

        Args:
            input_vocab_size (int): input vocabulary size
            target_vocab_size (int): target output vocabulary size
            seq_len_input (int): input sequence length
            seq_len_target (int): target output sequence length
            num_layers (int): number of layers
            d_model (int): input dimension
            num_heads (int): number of attention heads
            dff (int): inner layer dimension
            dropout_rate (float): dropout rate
            **kwargs: keyword arguments to be passed to super(Transformer, self).__init__()
        """
        super(Transformer, self).__init__(**kwargs)

        self.encoder = Encoder(
            num_layers, d_model, num_heads, dff,
            input_vocab_size, seq_len_input, dropout_rate)

        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff,
            target_vocab_size, seq_len_target, dropout_rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        """
        Implements the transformer model.

        Args:
            inp (tf.Tensor): input tensor
            tar (tf.Tensor): target tensor
            enc_padding_mask (tf.Tensor): encoding padding mask
            look_ahead_mask (tf.Tensor): look ahead mask

        Returns:
            final_output (tf.Tensor): output tensor, shape (batch_size, tar_seq_len, target_vocab_size)
            attention_weights (tf.Tensor): attention weights
        """

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(inp, enc_padding_mask)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask)

        # (batch_size, tar_seq_len, target_vocab_size)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        """
        Custom learning rate schedule for optimization.

        Args:
            d_model (int): input dimension
            warmup_steps (int): number of warmup steps
            **kwargs: keyword arguments to be passed to super(CustomSchedule, self).__init__()

        Returns:
            None
        """
        super(CustomSchedule, self).__init__(**kwargs)

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Implements the learning rate schedule

        Args:
            step (int): current step

        Returns:
            lr (tf.Tensor): learning rate
        """
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
