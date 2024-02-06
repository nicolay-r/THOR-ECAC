import argparse
import os
from collections import Counter
from os.path import dirname, join, realpath

import yaml
from attrdict import AttrDict

from src.service import CsvService, THoRFrameworkService, download


current_dir = dirname(realpath(__file__))
DATA_DIR = join(current_dir, "data")

DS_CAUSE_NAME = "cause_se24"
DS_CAUSE_DIR = join(DATA_DIR, DS_CAUSE_NAME)

DS_CAUSE_S1_NAME = "cause_se24_sparse_d1"
DS_CAUSE_S1_DIR = join(DATA_DIR, DS_CAUSE_NAME)

DS_STATE_NAME = "state_se24"
DS_STATE_DIR = join(DATA_DIR, DS_STATE_NAME)


CAUSE_FINAL_DATA = join(DS_CAUSE_DIR, "cause_final_en.csv")


def log_display_labels_stat(records_list):
    e_state = Counter()
    e_cause = Counter()
    for _, _, emotion_state, emotion_cause in records_list:
        e_state[config.label_list[emotion_state]] += 1
        e_cause[config.label_list[emotion_cause]] += 1
    print("Emotion State:", e_state)
    print("Emotion Cause:", e_cause)


def se24_cause(src, target):
    records_list = [[item[0], item[1], int(config.label_list.index(item[2])), int(config.label_list.index(item[3]))]
                    for item in CsvService.read(target=src, skip_header=True, cols=["context", "source", "emotion_state", "emotion_cause"])]

    no_label_uint = config.label_list.index(config.no_label)
    THoRFrameworkService.write_dataset(target_template=target, entries_it=records_list)
    print(f"No label: {no_label_uint}")
    log_display_labels_stat(records_list)
    print("---")


def se24_states(src, target):
    records_list = [[item[0], item[1], int(config.label_list.index(item[2])), int(config.label_list.index(config.no_label))]
                    for item in CsvService.read(target=src, skip_header=True, cols=["context", "target", "emotion"])]
    no_label_uint = config.label_list.index(config.no_label)
    THoRFrameworkService.write_dataset(target_template=target, entries_it=records_list)
    print(f"No label: {no_label_uint}")
    log_display_labels_stat(records_list)
    print("---")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--cause-train', dest="cause_train_data", type=str)
    parser.add_argument('--cause-valid', dest="cause_valid_data", type=str)
    parser.add_argument('--cause-test', dest="cause_test_data", type=str)
    parser.add_argument('--state-train', dest="state_train_data", type=str)
    parser.add_argument('--state-valid', dest="state_valid_data", type=str)
    parser.add_argument('--config', default='./config/config.yaml', help='config file')
    parser.add_argument('--skip-download', dest="skip_download", action='store_true', default=False)
    args = parser.parse_args()

    # Reading configuration.
    config = AttrDict(yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader))
    names = []
    for k, v in vars(args).items():
        setattr(config, k, v)

    data_sources = {
        join(DS_CAUSE_DIR, "cause_train_en.csv"): args.cause_train_data,
        join(DS_CAUSE_DIR, "cause_valid_en.csv"): args.cause_valid_data,
        CAUSE_FINAL_DATA: args.cause_test_data,
        join(DS_STATE_DIR, "state_train_en.csv"): args.state_train_data,
        join(DS_STATE_DIR, "state_valid_en.csv"): args.state_valid_data,
    }

    pickle_cause_se2024_data = {
        join(DS_CAUSE_DIR, f"{DS_CAUSE_NAME.capitalize()}_train"): join(DS_CAUSE_DIR, "cause_train_en.csv"),
        join(DS_CAUSE_DIR, f"{DS_CAUSE_NAME.capitalize()}_valid"): join(DS_CAUSE_DIR, "cause_valid_en.csv"),
        join(DS_CAUSE_DIR, f"{DS_CAUSE_NAME.capitalize()}_test"): join(DS_CAUSE_DIR, "cause_final_en.csv"),
    }

    pickle_state_se2024_data = {
        join(DS_STATE_DIR, f"{DS_STATE_NAME.capitalize()}_train"): join(DS_STATE_DIR, "state_train_en.csv"),
        join(DS_STATE_DIR, f"{DS_STATE_NAME.capitalize()}_valid"): join(DS_STATE_DIR, "state_valid_en.csv"),
        join(DS_STATE_DIR, f"{DS_STATE_NAME.capitalize()}_test"): join(DS_STATE_DIR, "state_valid_en.csv"),
    }

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not args.skip_download:
        for target, url in data_sources.items():
            download(dest_file_path=target, source_url=url)

    for target, src in pickle_cause_se2024_data.items():
        se24_cause(src, target)

    for target, src in pickle_state_se2024_data.items():
        se24_states(src, target)
