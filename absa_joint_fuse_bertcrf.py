'''
    bert + crfs for ner with tf2.0
    bert通过transformers加载
    一个crf 多个标注类型{O,T} 和 {O,POS,NEU,NEG}合并成一个{O,T,POS,NEU,NEG}
    计算loss时通过一个类别嵌入将任务指定为标注{O,T} 或 {O,POS,NEU,NEG}
'''

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.optimizers import Adam
from transformers.optimization_tf import AdamWeightDecay
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from absalayer import CRFuse

import sys, os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 屏蔽警告信息

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--batch_size', type=int, default=8, help='Batch size during training')
parser.add_argument('--epochs', type=int, default=10, help='Epochs during training')
parser.add_argument('--lr', type=float, default=1.0e-5, help='Initial learing rate')
parser.add_argument('--label_dim', type=int, default=5, help='number of ner labels')
parser.add_argument('--check', type=str, default='model/absa_joint_fuse_bertcrf',
                    help='The path where model shall be saved')
parser.add_argument('--mode', type=str, default='train0', help='The mode of train or predict as follows: '
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

    def call(self, sen, **kwargs):
        sequencemask = tf.cast(tf.greater(sen, 0), tf.int32)

        return tf.reduce_sum(sequencemask, axis=-1) - 2


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

    def build_model(self):
        sen = Input(shape=[None], name='sen', dtype=tf.int32)
        lab = Input(shape=[None], name='lab', dtype=tf.int32)

        seqlen = Mask(name="mask")(sen)

        sequence_output = BERT(name="bert")(sen)

        sequence_split = SplitSequence(name="splitsequence")(sequence_output)

        predict = CRFuse(label_dim=params.label_dim, name="crfuse")(inputs=(sequence_split, lab, seqlen))

        model = Model(inputs=[sen, lab], outputs=predict)

        model.summary()

        return model

    def train(self):
        model = self.build_model()

        if params.mode == 'train1':
            model.load_weights(params.check + '/absa.h5')

        optimizer = AdamWeightDecay(learning_rate=params.lr,
                                    weight_decay_rate=0.01,
                                    epsilon=1.0e-6,
                                    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

        model.compile(optimizer=optimizer)

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
            EarlyStopping(monitor='val_crf_acc', patience=3),
            ModelCheckpoint(filepath=params.check + '/absa.h5',
                            monitor='val_crf_acc',
                            save_best_only=True),
            CheckCallback()
        ]

        history = model.fit(batch_data,
                            epochs=params.epochs,
                            validation_data=test_data,
                            callbacks=callbacks
                            )

        with open(params.check + "/history.txt", "w", encoding="utf-8") as fw:
            fw.write(str(history.history))

    def predict(self):
        model = self.build_model()
        model.load_weights(params.check + '/absa.h5')

        predict = model.predict([sents, tf.ones_like(sents)[:, 1:-1]])

        querycheck(predict)


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
        user.train()
    else:
        user.predict()
