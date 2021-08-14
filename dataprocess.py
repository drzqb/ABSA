from transformers import BertTokenizer
import tensorflow as tf
from tqdm import tqdm


def laptop14(filepath, tfrecordfilepath):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    writer = tf.io.TFRecordWriter(tfrecordfilepath)

    label_dict = {"O": 0, "T-POS": 1, "T-NEU": 2, "T-NEG": 3}
    # label_dict_fix = {"O": 0, "B-POS": 1, "B-NEU": 2, "B-NEG": 3,"I-POS": 4, "I-NEU": 5, "I-NEG": 6,"S-POS": 7, "S-NEU": 8, "S-NEG": 9}

    m_samples = 0
    with open(filepath, "r", encoding="utf-8") as fr:
        for line in tqdm(fr):
            line = line.split("####")[-1]
            terms = line.strip().split(" ")

            sent2id = [101]
            label2id = []

            succeed = 1
            for term in terms:
                if "=" in term:
                    if "==" in term:
                        sent2id.append(tokenizer.encode("=", add_special_tokens=False)[0])
                        label2id.append(label_dict["O"])
                    else:
                        res = term.split("=")

                        res = [r for r in res if len(r) > 0]

                        if len(res) != 2:
                            succeed = 0
                            break

                        # print(term, " ---> ", res[0], "  ", res[1])

                        sentid = tokenizer.encode(res[0], add_special_tokens=False)
                        sent2id.extend(sentid)
                        label2id.extend([label_dict[res[1]]] * len(sentid))
                else:
                    succeed = 0
                    break
            if succeed == 0:
                print("ERROR")
                continue

            sent2id.append(102)

            assert len(sent2id) == len(label2id) + 2

            sen_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[sen_])) for sen_ in sent2id]

            lab_feature = [tf.train.Feature(int64_list=tf.train.Int64List(value=[lab_])) for lab_ in label2id]

            seq_example = tf.train.SequenceExample(
                feature_lists=tf.train.FeatureLists(feature_list={
                    'sen': tf.train.FeatureList(feature=sen_feature),
                    'lab': tf.train.FeatureList(feature=lab_feature)
                })
            )

            serialized = seq_example.SerializeToString()

            writer.write(serialized)

            m_samples += 1

    writer.close()

    print(m_samples)


if __name__ == "__main__":
    laptop14("data/OriginalFiles/laptop14_fix_train.txt", "data/TFRecordFiles/laptop14_train.tfrecord") # 3045
    laptop14("data/OriginalFiles/laptop14_fix_test.txt", "data/TFRecordFiles/laptop14_test.tfrecord")   # 800
