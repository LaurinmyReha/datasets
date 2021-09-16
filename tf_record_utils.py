# create tf record with normalized and downsampled waveform, samplerate, speaker id, clean text labels and other metadata
import utils
import tqdm
import tensorflow as tf
import os
def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    value = str(value)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature_list(value):
    """Returns a list of float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def int64_feature(value):
    value = int(value)
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def create_example_cv(waveform, path, text, samplerate, speakerid, up_votes, down_votes, age, gender, accent, locale,
                      segment):
    feature = {
        "waveform": float_feature_list(waveform),
        "path": bytes_feature(path),
        "text": bytes_feature(text),
        "samplerate": int64_feature(samplerate),
        "speakerid": bytes_feature(speakerid),
        "up_votes": int64_feature(up_votes),
        "down_votes": int64_feature(down_votes),
        "age": bytes_feature(age),
        "gender": bytes_feature(gender),
        "accent": bytes_feature(accent),
        "locale": bytes_feature(locale),
        "segment": bytes_feature(segment)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def create_example_mls(waveform, path, text, samplerate, speakerid, bookid, gender):
    feature = {
        "waveform": float_feature_list(waveform),
        "path": bytes_feature(path),
        "text": bytes_feature(text),
        "samplerate": int64_feature(samplerate),
        "speakerid": int64_feature(speakerid),
        "bookid": int64_feature(bookid),
        "gender": bytes_feature(gender)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example


def parse_tfrecord_fn_cv(example):
    feature_description = {
        "waveform": tf.io.VarLenFeature(tf.float32),
        "path": tf.io.VarLenFeature(tf.string),
        "text": tf.io.VarLenFeature(tf.string),
        "samplerate": tf.io.FixedLenFeature([], tf.int64),
        "speakerid": tf.io.VarLenFeature(tf.string),
        "up_votes": tf.io.FixedLenFeature([], tf.int64),
        "down_votes": tf.io.FixedLenFeature([], tf.int64),
        "age": tf.io.VarLenFeature(tf.string),
        "gender": tf.io.FixedLenFeature([], tf.string),
        "accent": tf.io.FixedLenFeature([], tf.string),
        "locale": tf.io.FixedLenFeature([], tf.string),
        "segment": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["age"] = tf.sparse.to_dense(example["age"])
    example["waveform"] = tf.sparse.to_dense(example["waveform"])
    example["speakerid"] = tf.sparse.to_dense(example["speakerid"])
    example["text"] = tf.sparse.to_dense(example["text"])
    example["path"] = tf.sparse.to_dense(example["path"])
    return example


def parse_tfrecord_fn_mls(example):
    feature_description = {
        "waveform": tf.io.VarLenFeature(tf.float32),
        "path": tf.io.VarLenFeature(tf.string),
        "text": tf.io.VarLenFeature(tf.string),
        "samplerate": tf.io.FixedLenFeature([], tf.int64),
        "speakerid": tf.io.VarLenFeature(tf.int64),
        "bookid": tf.io.VarLenFeature(tf.int64),
        "gender": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["waveform"] = tf.sparse.to_dense(example["waveform"])
    example["speakerid"] = tf.sparse.to_dense(example["speakerid"])
    example["text"] = tf.sparse.to_dense(example["text"])
    example["path"] = tf.sparse.to_dense(example["path"])
    example["bookid"] = tf.sparse.to_dense(example["bookid"])
    return example


def get_num_tf_records(dataset, num_samples_per_record):
    num_samples = dataset.length
    num_tf_records = num_samples // num_samples_per_record
    if num_samples % num_samples_per_record:
        num_tf_records += 1  # add one record if there are any remaining samples
    return num_tf_records


def generate_tf_records_cv(tf_records_dir, path_metadata, dataset, num_samples_per_record):
    if not os.path.exists(tf_records_dir):
        os.makedirs(tf_records_dir)

    num_tf_records = get_num_tf_records(dataset, num_samples_per_record)
    metadata = utils.load_tsv(path_metadata)

    for tfrec_num in tqdm(range(num_tf_records)):
        indexes = list(range(tfrec_num * num_samples_per_record, (tfrec_num + 1) * num_samples_per_record))
        preprocessed_batch = dataset.preprocess_batch(indexes)
        with tf.io.TFRecordWriter(
                tf_records_dir + "/tfrecord-%.5i-of-%.5i.tfrec" % (int(tfrec_num), int(num_tf_records))
        ) as writer:
            for index in indexes:
                audiofile = preprocessed_batch[index][0]
                path = dataset.paths[index]
                waveform = audiofile.waveform
                samplerate = audiofile.samplerate
                speakerid = metadata.loc[index, 'client_id']
                text = preprocessed_batch[index][1].text
                up_votes = metadata.loc[index, 'up_votes']
                down_votes = metadata.loc[index, 'down_votes']
                age = metadata.loc[index, 'age']
                gender = metadata.loc[index, 'gender']
                accent = metadata.loc[index, 'accent']
                locale = metadata.loc[index, 'locale']
                segment = metadata.loc[index, 'segment']
                example = create_example_cv(waveform, path, text, samplerate, speakerid, up_votes, down_votes, age,
                                            gender, accent, locale, segment)

                writer.write(example.SerializeToString())


def generate_tf_records_mls(tf_records_dir, path_metadata, dataset, num_samples_per_record):
    if not os.path.exists(tf_records_dir):
        os.makedirs(tf_records_dir)

    speaker_to_gender_dict = utils.get_speaker_gender_mls()
    num_tf_records = get_num_tf_records(dataset, num_samples_per_record)
    for tfrec_num in tqdm(range(num_tf_records)):
        indexes = list(range(tfrec_num * num_samples_per_record, (tfrec_num + 1) * num_samples_per_record))
        preprocessed_batch = dataset.preprocess_batch(indexes)
        with tf.io.TFRecordWriter(
                tf_records_dir + "/tfrecord-%.5i-of-%.5i.tfrec" % (int(tfrec_num), int(num_tf_records))
        ) as writer:
            for index in indexes:
                audiofile = preprocessed_batch[index][0]
                waveform = audiofile.waveform
                samplerate = audiofile.samplerate
                path = dataset.paths[index]
                speakerid = utils.get_book_and_speakerid_from_mls_path(dataset.paths[index])[0]
                bookid = utils.get_book_and_speakerid_from_mls_path(dataset.paths[index])[1]
                text = preprocessed_batch[index][1].text
                gender = speaker_to_gender_dict[int(speakerid)]
                example = create_example_mls(waveform, path, text, samplerate, speakerid, bookid, gender)
                writer.write(example.SerializeToString())

    return None


def get_tf_record_mls(path):
    raw_dataset = tf.data.TFRecordDataset(path)
    parsed_dataset = raw_dataset.map(parse_tfrecord_fn_mls)
    for features in parsed_dataset.take(1):

        for key in features.keys():
            print(key)
            print(f"{key}:{str(features[key])}")
    return parsed_dataset.take(1)


def generate_tf_records(tf_records_dir,path_metadata,dataset,num_samples_per_record):
    if(dataset.name=="MLS"):
        generate_tf_records_mls(tf_records_dir,path_metadata,dataset,num_samples_per_record)
    elif(dataset.name=="common_voice"):
        generate_tf_records_mls(tf_records_dir,path_metadata,dataset,num_samples_per_record)