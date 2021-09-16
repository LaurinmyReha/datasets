# This is a sample Python script.
import argparse
import dataset
import tf_record_utils as tfu
import sys
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
parser = argparse.ArgumentParser(description='Argument parser for processing speech datasets into serialized records')
parser.add_argument('--sample_rate', default=16000, type=int, help='The samplerate for the output')
parser.add_argument('--name_dataset', default="common_voice", type=str, help='The dataset ("common_voice" or "MLS"')
parser.add_argument('--language_code', default="de", type=str,
                    help='many datasets support many languages, the code specifies which dataset should be processed')
parser.add_argument('--root', default="/data", type=str, help='the root volume')
parser.add_argument('--split_size', default=550, type=int, help='how many records to pack into one serialized_record')
args = parser.parse_args()
print(args)
if (args.name_dataset == "common_voice"):
    path_labels = "cv-corpus-7.0-2021-07-21/de/train.tsv"
    path_metadata = "cv-corpus-7.0-2021-07-21/de/train.tsv"
    path_data = "cv-corpus-7.0-2021-07-21/de/clips/"
elif (args.name_dataset == "MLS"):
    path_data = "mls_german/train/audio"
    path_metadata = "mls_german/metainfo.txt"
    path_labels = "mls_german/train/transcripts.txt"
else:
    print("The given dataset is not supported")
    sys.exit(0)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':



    dataset=dataset.Dataset(args.name_dataset)
    all_chars = dataset.extract_all_chars()
    set_all_chars = set(all_chars)
    allowed_chars = [' ', 'ß', 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                     'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F',
                     'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    set_allowed_chars = set(allowed_chars)
    chars_to_remove = list(all_chars.difference(allowed_chars))
    chars_to_replace = ['ß']
    replacement = ['ss']

    #set preprocessing params
    dataset.chars_to_remove = chars_to_remove
    dataset.chars_to_replace = chars_to_replace
    dataset.replacement = replacement
    dataset.lowercase = True
    dataset.target_samplerate = args.samplerate
    dataset.normalize_audio = True
    size_per_tf_record = args.split_size  # ca 45 min


    tfu.generate_tf_records('test', path_metadata, dataset, args.split_size)
