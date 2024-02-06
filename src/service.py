import csv
import os
import pickle
import sys
from collections import Counter

import requests
from tqdm import tqdm


def save_labels_with_meta(pairs_path_csv, labels_pred_list, target_path_csv, meta_columns, label_columns):
    meta = list(CsvService.read(target=pairs_path_csv, skip_header=True, cols=meta_columns))
    assert(len(labels_pred_list) == len(meta))
    CsvService.write(target=target_path_csv, header=meta_columns + label_columns,
                     lines_it=[meta[i] + labels_pred_list[i] for i in range(len(labels_pred_list))])


def download(dest_file_path, source_url):
    print(('Downloading from {src} to {dest}'.format(src=source_url, dest=dest_file_path)))

    sys.stdout.flush()
    datapath = os.path.dirname(dest_file_path)

    if not os.path.exists(datapath):
        os.makedirs(datapath, mode=0o755)

    dest_file_path = os.path.abspath(dest_file_path)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with open(dest_file_path, 'wb') as f:
        pbar = tqdm(total=total_length, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=32 * 1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


class TxtService:

    @staticmethod
    def read_lines(filepath):
        print("Opening file: {}".format(filepath))
        with open(filepath, "r") as f:
            return [line.strip() for line in f.readlines()]


class THoRFrameworkService:

    @staticmethod
    def __write(target, content):
        print(f"Write: {target}")
        with open(target, 'wb') as f:
            pickle.dump(content, f)

    @staticmethod
    def write_dataset(target_template, entries_it):
        """ THoR-related service for sampling.
        """

        records = []
        for e in entries_it:
            assert(isinstance(e, list) and len(e) == 4)
            assert(isinstance(e[0], str))   # Context
            assert(isinstance(e[1], str))   # Target
            assert(isinstance(e[2], int))   # Label 1
            assert(isinstance(e[3], int))   # Label 2
            records.append(e)

        print(f"Records written: {len(records)}")
        THoRFrameworkService.__write(target=f"{target_template}.pkl", content=records)


class CsvService:

    @staticmethod
    def read(target, delimiter='\t', quotechar='"', skip_header=False, cols=None, return_row_ids=False):
        assert(isinstance(cols, list) or cols is None)

        header = None
        with open(target, newline='\n') as f:
            for row_id, row in enumerate(csv.reader(f, delimiter=delimiter, quotechar=quotechar)):
                if skip_header and row_id == 0:
                    header = row
                    continue

                # Determine the content we wish to return.
                if cols is None:
                    content = row
                else:
                    row_d = {header[col_name]: value for col_name, value in enumerate(row)}
                    content = [row_d[col_name] for col_name in cols]

                # Optionally attach row_id to the content.
                yield [row_id] + content if return_row_ids else content

    @staticmethod
    def write(target, lines_it, header=None, notify=True):
        assert(isinstance(header, list) or header is None)

        counter = Counter()
        with open(target, "w") as f:
            w = csv.writer(f, delimiter="\t", quotechar='"', quoting=csv.QUOTE_MINIMAL)

            if header is not None:
                w.writerow(header)

            for content in lines_it:
                w.writerow(content)
                counter["written"] += 1

        if notify:
            print(f"Saved: {target}")
            print("Total rows: {}".format(counter["written"]))
