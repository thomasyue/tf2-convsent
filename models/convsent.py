import tensorflow as tf


class CNNEncoder(tf.keras.layers.Layer):
    def __init__(self,
                 filter_shape,
                 pool_size,
                 vocab_size,
                 embed_dim,
                 dropout_rate,
                 **kwargs):
        super(CNNEncoder, self).__init__(**kwargs)
        self.filter_shape = filter_shape
        self.pool_size = pool_size
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.pool2d = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_size, padding='valid')

    def build(self, input_shape):
        self.filter = self.add_variable(
            "filter",
            shape=self.filter_shape,
            initializer=tf.initializers.GlorotUniform()
        )

        self.bias = self.add_variable(
            "bhid",
            shape=[self.filter_shape[-1]],
            initializer=tf.initializers.zeros()
        )

    def call(self, x, training=None):
        # [None, 48, 300] -> [None, 1, 48, 300]
        x = tf.expand_dims(x, axis=2)  # axis=1 or axis=2?
        x = self.dropout(x, training)

        # [None, 1, 48, 300] -> [None, 46, 1, 200]
        x = tf.nn.conv2d(x, self.filter, [1, 1], 'VALID')

        # tanh outperform relu
        x = tf.math.tanh(x + self.bias)

        # [None, 46, 1, 200] -> [None, 1, 1, 200]
        x = self.pool2d(x)

        # flattent into [None, 200] because lstm hidden states require 2dims
        return tf.squeeze(x)

    def get_config(self,):
        pass


class Attention(tf.keras.layers.Layer):
    def __init__(self,
                 units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, encoder_output, hidden_state, training=None):
        # [None, 600] -> [None, 1, 600]
        hidden_state = tf.expand_dims(hidden_state, axis=1)

        # formula #4 (Bahdanau's additive style)
        # [None, steps, 600] -> [None, steps, 1]
        score = self.V(tf.nn.tanh(self.W1(encoder_output) + self.W2(hidden_state)))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * encoder_output
        #         context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

    def get_config(self,):
        pass


class CustomLSTMLayer(tf.keras.layers.Layer):
    def __init__(self,
                 units):
        super(CustomLSTMLayer, self).__init__()
        self.units = units
        self.dense = tf.keras.layers.Dense(self.units)
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)

    def call(self, inputs, states, training=None):
        h0 = self.dense(states)
        h0 = tf.math.tanh(h0)
        init_states = [h0, tf.zeros_like(h0)]

        # [None, step, 600]
        lstm_output = self.lstm(inputs, initial_state=init_states)[0]

        # concat [None, 1, 600], [None, steps-1, 600]
        out = tf.concat([tf.expand_dims(h0, axis=1), lstm_output[:, :-1]], axis=1)

        return out

    def get_config(self,):
        pass


# class TransposeEmbedding(tf.keras.layers.Layer):
#     """"""
#     def __init__(self,
#                  hidden_dim,
#                  vocab_size,
#                  embed_dim,
#                  embedding_layer=None,
#                  activation=None,
#                  **kwargs):
#         super(TransposeEmbedding, self).__init__(**kwargs)
#         self.hidden_dim = hidden_dim
#         self.vocab_size = vocab_size
#         self.embed_dim = embed_dim
#         self.embedding_layer = embedding_layer
#         self.activation = tf.keras.activations.get(activation)
#
#     def build(self, input_shape):
#         self.transpose_weight = tf.transpose(self.embedding_layer.weights[0])
#
#         self.kernel = self.add_variable(
#             "Vhid",
#             shape=[self.hidden_dim, self.embed_dim],
#             initializer=tf.initializers.GlorotUniform(),
#             trainable=True
#         )
#
#         self.bias = self.add_variable(
#             "bhid",
#             shape=[self.vocab_size],
#             initializer=tf.initializers.zeros(),
#             trainable=True
#         )
#         self.built = True
#
#     def call(self, inputs, training=None):
#         # [600, 300]*[300, vocab_size] -> [600, vocab_size]
#         Vhid = tf.matmul(self.kernel, self.transpose_weight)
#
#         # [None, steps, 600]*[600, vocab_size] -> [None, steps, vocab_size]
#         if self.activation is not None:
#             output = self.activation(tf.matmul(inputs, Vhid) + self.bias)
#         else:
#             output = tf.matmul(inputs, Vhid) + self.bias
#
#         return output
#
#     def get_config(self):
#         pass


# class ConvSent(tf.keras.Model):
#     def __init__(self,
#                  vocab_size,
#                  embed_dim,
#                  hidden_dim,
#                  dropout_rate,
#                  feature_maps,
#                  filter_hs,
#                  **kwargs):
#         super(ConvSent, self).__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.hidden_dim = hidden_dim
#         self.vocab_size = vocab_size
#         self.dropout_rate = dropout_rate
#         self.feature_maps = feature_maps
#         img_h = 40 + 2 * (filter_hs[-1] - 1)
#
#         self.embed = tf.keras.layers.Embedding(
#             self.vocab_size, self.embed_dim
#         )
#
#         filters = []
#         pool_sizes = []
#         for filter_h in filter_hs:
#             filters.append((filter_h, 1, self.embed_dim, self.feature_maps))
#             pool_sizes.append((img_h - filter_h + 1, 1))
#
#         self.encoder_layers = []
#         for i in range(len(filters)):
#             conv_layer = CNNEncoder(
#                 filters[i], pool_sizes[i], self.vocab_size, self.embed_dim, self.dropout_rate
#             )
#             self.encoder_layers.append(conv_layer)
#
#         self.attention = Attention(self.hidden_dim)
#         self.decoder = CustomLSTMLayer(hidden_dim)
#         self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
#         self.transpose_embed = TransposeEmbedding(
#             self.hidden_dim, self.vocab_size, self.embed_dim, self.embed
#         )
#
#     def call(self, x, y, training=None):
#         # -Encode-
#         x_embed = self.embed(x)
#
#         encoder_output = []
#         for layer in self.encoder_layers:
#             conv_output = layer(x_embed, training)
#             encoder_output.append(conv_output)
#
#         # 3*[None, 200] -> [None, 600]
#         latent_vector = tf.concat(encoder_output, axis=-1)
#         latent_vector = self.dropout(latent_vector, training)
#
#         # -Decode-
#         y_embed = self.embed(y)
#
#         # # attention
#         # context_vector, attention_weights = self.attention(y_embed, latent_vector)
#         # y_embed = tf.concat([context_vector, y_embed], axis=-1)
#
#         # y_embed=[None, steps, 300], h=[None, 600] -> [None, steps, 600]
#         pred = self.decoder(y_embed, latent_vector, training)
#
#         # extract word embedding matrix + fully connected layer
#         # [None, step, 300] * [300, vocab_size] -> [None, step, vocab_size]
#         pred_w = self.transpose_embed(pred)
#
#         return pred_w
#
#
#     def get_config(self,):
#         pass


class ConvSent(tf.keras.Model):
    def __init__(self,
                 vocab_size,
                 embed_dim,
                 hidden_dim,
                 dropout_rate,
                 feature_maps,
                 filter_hs,
                 embed_matrix=None,
                 **kwargs):
        super(ConvSent, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.feature_maps = feature_maps
        img_h = 40 + 2 * (filter_hs[-1] - 1)

        if embed_matrix is not None:
            self.embed = tf.keras.layers.Embedding(
                self.vocab_size, self.embed_dim, weights=[embed_matrix], trainable=True
            )
        else:
            self.embed = tf.keras.layers.Embedding(
                self.vocab_size, self.embed_dim
            )

        filters = []
        pool_sizes = []
        for filter_h in filter_hs:
            filters.append((filter_h, 1, self.embed_dim, self.feature_maps))
            pool_sizes.append((img_h - filter_h + 1, 1))

        self.encoder_layers = []
        for i in range(len(filters)):
            conv_layer = CNNEncoder(
                filters[i], pool_sizes[i], self.vocab_size, self.embed_dim, self.dropout_rate
            )
            self.encoder_layers.append(conv_layer)

        self.attention = Attention(self.hidden_dim)
        self.decoder = CustomLSTMLayer(hidden_dim)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.kernel = self.add_variable(
            "Vhid",
            shape=[self.hidden_dim, self.embed_dim],
            initializer=tf.initializers.GlorotUniform(),
            trainable=True
        )
        self.bias = self.add_variable(
            "bhid",
            shape=[self.vocab_size],
            initializer=tf.initializers.zeros(),
            trainable=True
        )

    def call(self, x, y, training=None):
        # -Encode-
        x_embed = self.embed(x)

        encoder_output = []
        for layer in self.encoder_layers:
            conv_output = layer(x_embed, training)
            encoder_output.append(conv_output)

        # 3*[None, 200] -> [None, 600]
        latent_vector = tf.concat(encoder_output, axis=-1)
        latent_vector = self.dropout(latent_vector, training)

        # -Decode-
        y_embed = self.embed(y)

        # # attention
        # context_vector, attention_weights = self.attention(y_embed, latent_vector)
        # y_embed = tf.concat([context_vector, y_embed], axis=-1)

        # y_embed=[None, steps, 300], h=[None, 600] -> [None, steps, 600]
        pred = self.decoder(y_embed, latent_vector, training)

        embed_matrix = tf.transpose(self.embed.weights[0])

        # extract word embedding matrix + fully connected layer
        # [None, step, 300] * [300, vocab_size] -> [None, step, vocab_size]
        Vhid = tf.matmul(self.kernel, embed_matrix)
        pred_w = tf.matmul(pred, Vhid) + self.bias

        return pred_w


    def get_config(self,):
        pass