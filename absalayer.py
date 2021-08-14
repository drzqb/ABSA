import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow_addons as tfa


class CRF(Layer):
    def __init__(self, label_dim, **kwargs):
        super(CRF, self).__init__(**kwargs)
        self.label_dim = label_dim

        self.dense_ner = Dense(label_dim,
                               kernel_initializer=TruncatedNormal(stddev=0.02),
                               dtype=tf.float32,
                               name='ner')

        self.transitions = self.add_weight(name='transitions',
                                           shape=[label_dim, label_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)

    def get_config(self):
        config = super(CRF, self).get_config().copy()
        config.update({
            'label_dim': self.label_dim,
        })

        return config

    def call(self, inputs, **kwargs):
        x, label, seqlen = inputs

        output = self.dense_ner(x)

        log_likelihood, _ = tfa.text.crf_log_likelihood(output, label, seqlen, self.transitions)
        loss = tf.reduce_mean(-log_likelihood)

        self.add_loss(loss)

        viterbi_sequence, _ = tfa.text.crf_decode(output, self.transitions, seqlen)

        accuracy = tf.reduce_sum(tf.cast(tf.equal(viterbi_sequence, label), tf.float32) * tf.cast(
            tf.sequence_mask(seqlen, tf.reduce_max(seqlen)), tf.float32))
        accuracy = accuracy / tf.reduce_sum(tf.cast(seqlen, tf.float32))

        self.add_metric(accuracy, name="crf_acc")

        return viterbi_sequence


class Linear(Layer):
    def __init__(self, label_dim, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.label_dim = label_dim

        self.dense_ner = Dense(label_dim,
                               kernel_initializer=TruncatedNormal(stddev=0.02),
                               dtype=tf.float32,
                               name='ner')

    def get_config(self):
        config = super(Linear, self).get_config().copy()
        config.update({
            'label_dim': self.label_dim,
        })

        return config

    def call(self, inputs, **kwargs):
        x, label, seqlen = inputs

        sequence_mask = tf.sequence_mask(seqlen, tf.reduce_max(seqlen))

        seqlen_sum = tf.cast(tf.reduce_sum(seqlen), tf.float32)

        output = self.dense_ner(x)

        loss = tf.keras.losses.sparse_categorical_crossentropy(label, output, from_logits=True)
        lossf = tf.zeros_like(loss)
        loss = tf.where(sequence_mask, loss, lossf)

        self.add_loss(tf.reduce_sum(loss) / seqlen_sum)

        predict = tf.argmax(output, axis=-1, output_type=tf.int32)

        accuracy = tf.cast(tf.equal(predict, label), tf.float32)
        accuracyf = tf.zeros_like(accuracy)
        accuracy = tf.where(sequence_mask, accuracy, accuracyf)

        self.add_metric(tf.reduce_sum(accuracy) / seqlen_sum, name="acc")

        return predict
