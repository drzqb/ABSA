'''
    bert + crfs for ner with tf2.0
    bert通过transformers加载
    多个crf串联分别预测不同的类别
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.optimizers import Adam
from transformers.optimization_tf import AdamWeightDecay
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from absalayer import CRF, Fuse

import sys, os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=10, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--label_dim', type=int, default=[2, 4], help='number of ner labels')
parser.add_argument('--check', type=str, default='model/absa_joint_series_bertcrf',
                    help='The path where model shall be saved')
parser.add_argument('--mode', type=str, default='test', help='The mode of train or predict as follows: '
                                                             'train0: begin to train or retrain'
                                                             'tran1:continue to train'
                                                             'predict: predict')
params = parser.parse_args()


def single_example_parser(serialized_example):
    sequence_features = {
        'sen': tf.io.FixedLenSequenceFeature([], tf.int64),
        'lab': tf.io.FixedLenSequenceFeature([], tf.int64),
    }

    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=serialized_example,
        sequence_features=sequence_features
    )

    sen = sequence_parsed['sen']
    lab = sequence_parsed['lab']
    return {"sen": sen, "lab": lab}


def batched_data(tfrecord_filename, single_example_parser, batch_size, padded_shapes, buffer_size=1000, shuffle=True):
    dataset = tf.data.TFRecordDataset(tfrecord_filename)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.map(single_example_parser) \
        .padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=True) \
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


class Mask(Layer):
    def __init__(self, **kwargs):
        super(Mask, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        sen, label = inputs
        sequencemask = tf.cast(tf.greater(sen, 0), tf.int32)

        label1 = tf.where(tf.greater(label, 0), tf.ones_like(label), tf.zeros_like(label))
        label2 = 1 * label

        return tf.reduce_sum(sequencemask, axis=-1) - 2, label1, label2


class BERT(Layer):
    def __init__(self, **kwargs):
        super(BERT, self).__init__(**kwargs)

        self.bert = TFBertModel.from_pretrained("bert-base-uncased")

    def call(self, inputs, **kwargs):
        return self.bert(inputs)[0]


class SplitSequence(Layer):
    def __init__(self, **kwargs):
        super(SplitSequence, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        return x[:, 1:-1]


class CheckCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        predict = self.model.predict([sents, tf.ones_like(sents)[:, 1:-1]])
        querycheck(predict)


def querycheck(predict):
    sys.stdout.write('\n')
    sys.stdout.flush()
    for i, pre in enumerate(predict):
        sys.stdout.write(newsentences[i] + '\n')
        for j in range(lens[i]):
            sys.stdout.write(ner_inverse_dict[pre[j]] + ' ')
        sys.stdout.write('\n\n')
        sys.stdout.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()


class USER:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def build_T_model(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)
        lab = Input(shape=[None], name='lab', dtype=tf.int32)

        seqlen, label1, _ = Mask(name="mask")(inputs=(sen, lab))

        sequence_output = BERT(name="bert")(sen)

        sequence_split = SplitSequence(name="splitsequence")(sequence_output)

        predict = CRF(label_dim=params.label_dim[0], name="crf")(inputs=(sequence_split, label1, seqlen))

        model = Model(inputs=[sen, lab], outputs=predict)

        model.summary()

        return model

    def build_Senti_model(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)
        lab = Input(shape=[None], name='lab', dtype=tf.int32)

        seqlen, _, label2 = Mask(name="mask")(inputs=(sen, lab))

        sequence_output = BERT(name="bert")(sen)

        sequence_split = SplitSequence(name="splitsequence")(sequence_output)

        predict = CRF(label_dim=params.label_dim[1], name="crf")(inputs=(sequence_split, label2, seqlen))

        model = Model(inputs=[sen, lab], outputs=predict)

        model.summary()

        return model

    def build_fuse_model(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)
        lab = Input(shape=[None], name='lab', dtype=tf.int32)

        seqlen, _, _ = Mask(name="mask")(inputs=(sen, lab))

        modelT = self.build_T_model()
        modelT.load_weights(params.check + '/absaT.h5')
        predictT = modelT([sen, lab])

        modelSenti = self.build_Senti_model()
        modelSenti.load_weights(params.check + '/absaSenti.h5')

        predictSenti = modelSenti([sen, lab])

        predict = Fuse(name="fuse")(inputs=(predictT, predictSenti, lab, seqlen))

        model = Model(inputs=[sen, lab], outputs=predict)

        model.summary()

        return model

    def trainT(self):
        modelT = self.build_T_model()

        if params.mode == 'train1':
            modelT.load_weights(params.check + '/absaT.h5')

        optimizer = AdamWeightDecay(learning_rate=params.lr,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        modelT.compile(optimizer=optimizer)

        batch_data = batched_data(['data/TFRecordFiles/laptop14_train.tfrecord'],
                                  single_example_parser,
                                  params.batch_size,
                                  padded_shapes={"sen": [-1], "lab": [-1]},
                                  buffer_size=100 * params.batch_size)

        test_data = batched_data(['data/TFRecordFiles/laptop14_test.tfrecord'],
                                 single_example_parser,
                                 params.batch_size,
                                 padded_shapes={"sen": [-1], "lab": [-1]},
                                 buffer_size=100 * params.batch_size)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3),
            ModelCheckpoint(filepath=params.check + '/absaT.h5',
                            monitor='val_loss',
                            save_best_only=True),
        ]

        history = modelT.fit(batch_data,
                             epochs=params.epochs,
                             validation_data=test_data,
                             callbacks=callbacks
                             )

        with open(params.check + "/historyT.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history.history))

    def trainSenti(self):
        modelSenti = self.build_Senti_model()

        if params.mode == 'train1':
            modelSenti.load_weights(params.check + '/absaSenti.h5')

        optimizer = AdamWeightDecay(learning_rate=params.lr,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        modelSenti.compile(optimizer=optimizer)

        batch_data = batched_data(['data/TFRecordFiles/laptop14_train.tfrecord'],
                                  single_example_parser,
                                  params.batch_size,
                                  padded_shapes={"sen": [-1], "lab": [-1]},
                                  buffer_size=100 * params.batch_size)

        test_data = batched_data(['data/TFRecordFiles/laptop14_test.tfrecord'],
                                 single_example_parser,
                                 params.batch_size,
                                 padded_shapes={"sen": [-1], "lab": [-1]},
                                 buffer_size=100 * params.batch_size)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3),
            ModelCheckpoint(filepath=params.check + '/absaSenti.h5',
                            monitor='val_loss',
                            save_best_only=True),
        ]

        history = modelSenti.fit(batch_data,
                                 epochs=params.epochs,
                                 validation_data=test_data,
                                 callbacks=callbacks
                                 )

        with open(params.check + "/historySenti.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history.history))

    def predict(self):
        model = self.build_fuse_model()
        predict = model.predict([sents, tf.ones_like(sents)[:, 1:-1]])

        querycheck(predict)

    def test(self):
        model = self.build_fuse_model()
        model.compile()

        test_data = batched_data(['data/TFRecordFiles/laptop14_test.tfrecord'],
                                 single_example_parser,
                                 params.batch_size,
                                 padded_shapes={"sen": [-1], "lab": [-1]},
                                 buffer_size=100 * params.batch_size)

        model.evaluate(test_data)


if __name__ == '__main__':
    ner_dict = {"O": 0, "T-POS": 1, "T-NEU": 2, "T-NEG": 3}
    ner_inverse_dict = {v: k for k, v in ner_dict.items()}

    if not os.path.exists(params.check):
        os.makedirs(params.check)

    user = USER()
    sentences = [
        'MAYBE The Mac OS improvement were not The product they Want to offer.',
        'They are simpler to use.',
        'so I called technical support.',
    ]

    newsentences = [user.tokenizer.tokenize(sentences[i]) for i in range(len(sentences))]
    lens = [len(newsentences[i]) for i in range(len(sentences))]

    newsentences = [" ".join(newsentences[i]) for i in range(len(sentences))]

    sents = user.tokenizer(sentences, padding=True, return_tensors="tf")["input_ids"]

    if params.mode.startswith('train'):
        user.trainT()
        user.trainSenti()
    elif params.mode == "predict":
        user.predict()
    elif params.mode == "test":
        user.test()
