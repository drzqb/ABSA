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


class CRFuse(Layer):
    def __init__(self, label_dim, **kwargs):
        super(CRFuse, self).__init__(**kwargs)
        self.label_dim = label_dim

        self.dense_ner = Dense(label_dim,
                               kernel_initializer=TruncatedNormal(stddev=0.02),
                               dtype=tf.float32,
                               name='ner')

        self.cls_embed = self.add_weight(name="cls_embed",
                                         shape=[2, 768],
                                         dtype=tf.float32,
                                         trainable=True)

        self.transitions = self.add_weight(name='transitions',
                                           shape=[label_dim, label_dim],
                                           initializer='glorot_uniform',
                                           trainable=True)

    def get_config(self):
        config = super(CRFuse, self).get_config().copy()
        config.update({
            'label_dim': self.label_dim,
        })

        return config

    def call(self, inputs, **kwargs):
        x, label, seqlen = inputs

        x1 = x + self.cls_embed[0]
        x2 = x + self.cls_embed[1]
        x_r = tf.concat([x1, x2], axis=0)
        output_r = self.dense_ner(x_r)
        # output1 = output + self.cls_embed[0]
        # output2 = output + self.cls_embed[1]
        # output_r = tf.concat([output1, output2], axis=0)

        label1 = tf.where(tf.greater(label, 0), tf.ones_like(label), tf.zeros_like(label))
        label2 = tf.where(tf.greater(label, 0), label + 1, tf.zeros_like(label))
        label_r = tf.concat([label1, label2], axis=0)

        seqlen_r = tf.tile(seqlen, [2])

        log_likelihood, _ = tfa.text.crf_log_likelihood(output_r, label_r, seqlen_r, self.transitions)
        loss = tf.reduce_mean(-log_likelihood)

        self.add_loss(loss)

        viterbi_sequence_r, _ = tfa.text.crf_decode(output_r, self.transitions, seqlen_r)

        viterbi_sequence1, viterbi_sequence2 = tf.split(viterbi_sequence_r, 2)

        viterbi_sequence = tf.where(tf.logical_or(tf.equal(viterbi_sequence1, 0), tf.equal(viterbi_sequence2, 0)),
                                    tf.zeros_like(viterbi_sequence1),
                                    viterbi_sequence2 - 1)
        accuracy = tf.reduce_sum(tf.cast(tf.equal(viterbi_sequence, label), tf.float32) * tf.cast(
            tf.sequence_mask(seqlen, tf.reduce_max(seqlen)), tf.float32))
        accuracy = accuracy / tf.reduce_sum(tf.cast(seqlen, tf.float32))

        self.add_metric(accuracy, name="crf_acc")

        return viterbi_sequence


class CRFMRC(Layer):
    def __init__(self, label_dim, **kwargs):
        super(CRFMRC, self).__init__(**kwargs)
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
        config = super(CRFMRC, self).get_config().copy()
        config.update({
            'label_dim': self.label_dim,
        })

        return config

    def call(self, inputs, **kwargs):
        x, label, seqlen = inputs

        output_r = self.dense_ner(x)

        label1 = tf.where(tf.greater(label, 0), tf.ones_like(label), label)
        label2 = tf.where(tf.greater(label, 0), label + 1, label)
        label_r = tf.concat([label1, label2], axis=0)

        seqlen_r = tf.tile(seqlen, [2])

        log_likelihood, _ = tfa.text.crf_log_likelihood(output_r, label_r, seqlen_r, self.transitions)
        loss = tf.reduce_mean(-log_likelihood)

        self.add_loss(loss)

        viterbi_sequence_r, _ = tfa.text.crf_decode(output_r, self.transitions, seqlen_r)

        viterbi_sequence1, viterbi_sequence2 = tf.split(viterbi_sequence_r, 2)

        viterbi_sequence = tf.where(tf.logical_or(tf.equal(viterbi_sequence1, 0), tf.equal(viterbi_sequence2, 0)),
                                    tf.zeros_like(viterbi_sequence1),
                                    viterbi_sequence2 - 1)
        accuracy = tf.reduce_sum(tf.cast(tf.equal(viterbi_sequence, label), tf.float32) * tf.cast(
            tf.sequence_mask(seqlen, tf.reduce_max(seqlen)), tf.float32))
        accuracy = accuracy / tf.reduce_sum(tf.cast(seqlen, tf.float32))

        self.add_metric(accuracy, name="crf_acc")

        return viterbi_sequence


class CRFs(Layer):
    def __init__(self, label_dim, **kwargs):
        super(CRFs, self).__init__(**kwargs)
        self.label_dim = label_dim  # [2,4]

    def build(self, input_shape):
        self.dense_ners = [Dense(self.label_dim[i],
                                 kernel_initializer=TruncatedNormal(stddev=0.02),
                                 dtype=tf.float32,
                                 name='ner' + str(i)) for i in range(2)]

        self.transitions = [self.add_weight(name='transitions' + str(i),
                                            shape=[self.label_dim[i], self.label_dim[i]],
                                            initializer='glorot_uniform',
                                            trainable=True) for i in range(2)]
        super(CRFs, self).build(input_shape)

    def get_config(self):
        config = super(CRFs, self).get_config().copy()
        config.update({
            'label_dim': self.label_dim,
        })

        return config

    def call(self, inputs, **kwargs):
        x, label, seqlen = inputs

        label1 = tf.where(tf.greater(label, 0), tf.ones_like(label), tf.zeros_like(label))

        output1 = self.dense_ners[0](x)

        log_likelihood1, _ = tfa.text.crf_log_likelihood(output1, label1, seqlen, self.transitions[0])
        loss1 = tf.reduce_mean(-log_likelihood1)

        viterbi_sequence1, _ = tfa.text.crf_decode(output1, self.transitions[0], seqlen)

        accuracy1 = tf.reduce_sum(tf.cast(tf.equal(viterbi_sequence1, label1), tf.float32) * tf.cast(
            tf.sequence_mask(seqlen, tf.reduce_max(seqlen)), tf.float32))
        accuracy1 = accuracy1 / tf.reduce_sum(tf.cast(seqlen, tf.float32))

        self.add_metric(accuracy1, name="crf_acc1")

        label2 = 1 * label

        output2 = self.dense_ners[1](x)

        log_likelihood2, _ = tfa.text.crf_log_likelihood(output2, label2, seqlen, self.transitions[1])
        loss2 = tf.reduce_mean(-log_likelihood2)

        viterbi_sequence2, _ = tfa.text.crf_decode(output2, self.transitions[1], seqlen)

        accuracy2 = tf.reduce_sum(tf.cast(tf.equal(viterbi_sequence2, label2), tf.float32) * tf.cast(
            tf.sequence_mask(seqlen, tf.reduce_max(seqlen)), tf.float32))
        accuracy2 = accuracy2 / tf.reduce_sum(tf.cast(seqlen, tf.float32))

        self.add_metric(accuracy2, name="crf_acc2")

        self.add_loss((loss1 + 8.0 * loss2) / 9.0)

        viterbi_sequence = tf.where(tf.logical_or(tf.equal(viterbi_sequence1, 0), tf.equal(viterbi_sequence1, 0)),
                                    tf.zeros_like(viterbi_sequence1), viterbi_sequence2)
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


class Fuse(Layer):
    def __init__(self, **kwargs):
        super(Fuse, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        predict1, predict2, label, seqlen = inputs

        predict = tf.where(tf.logical_or(tf.equal(predict1, 0), tf.equal(predict1, 0)),
                           tf.zeros_like(predict1), predict2)

        accuracy = tf.reduce_sum(tf.cast(tf.equal(predict, label), tf.float32) * tf.cast(
            tf.sequence_mask(seqlen, tf.reduce_max(seqlen)), tf.float32))
        accuracy = accuracy / tf.reduce_sum(tf.cast(seqlen, tf.float32))
        self.add_metric(accuracy, name="crf_acc")

        return predict
