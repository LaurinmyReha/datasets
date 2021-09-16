import pandas as pd
import os
import re

from main import path_data, path_metadata,path_labels
# needs refactoring
def get_paths_dataset(dataset="common_voice"):
    if (dataset == "common_voice"):
        return get_audio_paths_cv()

    elif (dataset == "MLS"):
        return get_audio_paths_mls()

    else:
        print("dataset not found")


def get_transcripts_dataset(dataset="common_voice"):
    if (dataset == "common_voice"):
        return get_transcripts_cv(path_labels)

    elif (dataset == "MLS"):
        return get_transcripts_mls(path_labels)

    else:
        print("dataset not found")


def get_transcripts_cv(path_labels):
    file_name_to_transcript_dict = {}
    audio_labels_df = load_tsv(path_labels)
    file_names = audio_labels_df["path"].tolist()
    labels = audio_labels_df["sentence"].tolist()
    for i, file_name in enumerate(file_names):
        file_name_to_transcript_dict[file_name] = labels[i]

    return file_name_to_transcript_dict


def get_transcripts_mls(path_labels):
    transcripts = pd.read_csv(path_labels, sep="\t", header=None)
    file_name_to_transcript_dict = {}
    for i in range(transcripts.shape[0]):
        x = list(transcripts.loc[i])
        file_name_to_transcript_dict[x[0] + '.flac'] = x[1].strip()
    return file_name_to_transcript_dict


def get_transcripts(datasetname: str):
    if (datasetname == "common_voice"):
        transcripts = get_transcripts_cv(path_labels)
    elif (datasetname == "MLS"):
        transcripts = get_transcripts_mls(path_labels)
    return transcripts

    return audio_labels_df["sentence"].tolist()


def get_audio_paths(datasetname: str):
    if (datasetname == "common_voice"):
        audio_paths = get_audio_paths_cv(path_labels)
    elif (datasetname == "MLS"):
        audio_paths = get_audio_paths_mls(path_data)
    return audio_paths


def get_audio_paths_mls():
    mls_train_data_audio_paths = []
    for root, subdirs, files in os.walk(path_data):
        for filename in files:
            file_path = os.path.join(root, filename)
            if ('flac' in file_path):
                mls_train_data_audio_paths.append(file_path)
    return mls_train_data_audio_paths


def get_audio_paths_cv():
    audio_labels_df = load_tsv(path_labels)
    file_names = audio_labels_df["path"].tolist()
    return [path_data + file_name for file_name in file_names]


def get_speaker_gender_mls():
    speaker_to_gender_dict = {}
    metadata_df = pd.read_csv(path_metadata, sep='|')
    for i in range(metadata_df.shape[0]):
        x = list(metadata_df.loc[i])
        speaker_to_gender_dict[x[0]] = x[1].strip()
    return speaker_to_gender_dict


def get_book_and_speakerid_from_mls_path(path):
    expression = re.compile('.*\/(.*)_(.*)_(.*).flac')
    m = expression.match(path)
    # [speaker_Id, bookId_recordingId]
    return [int(m.group(1)), int(m.group(2)), int(m.group(3))]


def get_filename_from_file_path(path):
    expression = re.compile('.*\/(.*)')
    m = expression.match(path)
    return m.group(1)


def load_tsv(path_labels: str):
    tsv_data = pd.read_csv(path_labels, sep='\t')
    audio_labels_df = tsv_data.sort_values("path", ignore_index=True)
    return audio_labels_df

