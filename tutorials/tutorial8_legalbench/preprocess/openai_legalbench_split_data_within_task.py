# Copyright © 2024 Reexpress AI, Inc.
# See the Reexpress AI tutorial for usage instructions.

"""
Split the data, regardless of task.
"""

import argparse
import json
import codecs
import time
import numpy as np
import glob
import os
import random


def get_data(filename_with_path):
    """
    Get the preprocessed LegalBench data
    :param filename_with_path: A filepath to the LegalBench preprocessed data. See the Tutorial for details.
    :return: A list of dictionaries
    """
    json_list = []
    with codecs.open(filename_with_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            json_list.append(json_obj)
    return json_list


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data.")

    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--input_dir", required=True, help="")
    parser.add_argument(
        "--output_jsonl_dir", default="", help="")

    args = parser.parse_args()

    random.seed(args.seed)
    filenames = glob.glob(f"{args.input_dir}/*.jsonl")
    print(f"Total tasks: {len(filenames)}")

    all_data = []
    for input_filename in filenames:
        input_data_json_list = get_data(input_filename)
        all_data.extend(input_data_json_list)

    random.shuffle(all_data)

    # (The percentage split in the tutorial is arbitrary, with 20% as eval.
    # Feel free to experiment with other splits to examine the behavior of Reexpress over varying data size splits.
    # We include a link to the data in the repo with the attributes. You can also examine the behavior when
    # splitting by task with openai_legalbench_split_data_by_task.py)
    eval_documents = all_data[0:int(len(all_data)*0.2)]
    train_documents = all_data[int(len(all_data)*0.2):int(len(all_data)*0.5)]
    calibration_documents = all_data[int(len(all_data)*0.5):]

    # Reexpress uses the 'id' as the unique key, so there is no chance of overlap in practice after importing
    # the data. Here we make sure there
    # is no overlap in case you want to use the files in other contexts:
    assert len(set([x["id"] for x in eval_documents]) &
               set([x["id"] for x in train_documents]) &
               set([x["id"] for x in calibration_documents])) == 0
    print(f"Eval docs: {len(eval_documents)}")
    print(f"Train docs: {len(train_documents)}")
    print(f"Calibration docs: {len(calibration_documents)}")

    save_json_lines(os.path.join(args.output_jsonl_dir, f'eval.legalbench.jsonl'), eval_documents)
    save_json_lines(os.path.join(args.output_jsonl_dir, f'train.legalbench.jsonl'), train_documents)
    save_json_lines(os.path.join(args.output_jsonl_dir, f'calibration.legalbench.jsonl'), calibration_documents)