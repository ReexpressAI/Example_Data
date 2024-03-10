# Copyright © 2024 Reexpress AI, Inc.
# See the Reexpress AI tutorial for usage instructions.

"""
Split data by task (i.e., keep all documents within a task in the same datasplit). You can use this to explore
the behavior of Reexpress on distribution shifts (as in Tutorial 7). Typically in real-world deployments,
especially for professional and enterprise tasks, examples from the target task (i.e., that for which you will be
producing predictions for new documents) should appear in both the Training and Calibration sets. In the tutorial,
we split the data regardless of task to emulate such a setting, using openai_legalbench_split_data_within_task.py.
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


def save_split(output_jsonl_dir, split_filenames, split_name):
    split = []
    for input_filename in split_filenames:
        input_data_json_list = get_data(input_filename)
        split.extend(input_data_json_list)

    save_json_lines(os.path.join(output_jsonl_dir, f'{split_name}.legalbench.jsonl'), split)
    print(f"Saved {split_name} with {len(split)} documents across {len(split_filenames)} tasks.")


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data.")

    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument(
        "--input_dir", required=True,
        help="")
    parser.add_argument(
        "--output_jsonl_dir", default="",
        help="")

    args = parser.parse_args()

    random.seed(args.seed)
    filenames = glob.glob(f"{args.input_dir}/*.jsonl")
    random.shuffle(filenames)
    print(f"Total tasks: {len(filenames)}")
    # 20% as eval:
    eval_filenames = filenames[0:int(len(filenames)*0.2)]
    # remaining split 50-50 into train and calibration:
    remaining_filenames = filenames[int(len(filenames)*0.2):]
    train_filenames = remaining_filenames[0:int(len(remaining_filenames)*0.5)]
    calibration_filenames = remaining_filenames[int(len(remaining_filenames)*0.5):]

    save_split(args.output_jsonl_dir, eval_filenames, "eval")
    save_split(args.output_jsonl_dir, train_filenames, "train")
    save_split(args.output_jsonl_dir, calibration_filenames, "calibration")



