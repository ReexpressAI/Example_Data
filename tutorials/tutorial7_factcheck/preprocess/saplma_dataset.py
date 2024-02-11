"""
This simply converts the csv-formatted data from the paper "The Internal State of an LLM Knows When It’s Lying"
(https://arxiv.org/abs/2304.13734) into the Reexpress JSON lines format.

"""

import argparse
import codecs
import json
import uuid
import random
from os import path
import csv


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def get_data(filepath, filename):
    json_list = []

    filename_with_path = path.join(filepath, filename)
    correct_doc_count = 0
    error_doc_count = 0
    line_index = 0
    with open(filename_with_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            assert len(row) == 2
            if line_index != 0:
                document = row[0]
                label = int(row[1])
                if label == 0:
                    error_doc_count += 1
                elif label == 1:
                    correct_doc_count += 1
                else:
                    assert False

                line_index_string = "{:06d}".format(line_index)
                json_obj = {"id": str(uuid.uuid4()), "label": label,
                            "document": document,
                            "info": f"{filename}: {line_index_string}"}
                json_list.append(json_obj)
            line_index += 1

    print(f"Correct: {correct_doc_count}, error: {error_doc_count}")
    return json_list


def get_label_display_names(string2label):
    json_list_labels = []
    for label_string in string2label:
        json_obj = {"label": string2label[label_string], "name": label_string}
        json_list_labels.append(json_obj)
    return json_list_labels


def main():
    parser = argparse.ArgumentParser(
        description="-----[Output JSON lines format.]-----")
    parser.add_argument(
        "--input_data_path", required=True,
        help="Data from https://arxiv.org/pdf/azariaa.com/Content/Datasets/true-false-dataset.zip")
    parser.add_argument(
        "--input_data_file", required=True,
        help="JSON lines output file. Must have the ending .jsonl")
    parser.add_argument(
        "--output_jsonl_file", default="",
        help="JSON lines output file. Must have the ending .jsonl")
    parser.add_argument(
        "--output_label_display_names_jsonl_file", default="",
        help="JSON lines output file for label display names. Must have the ending .jsonl")

    options = parser.parse_args()
    random.seed(0)

    json_list = get_data(options.input_data_path, options.input_data_file)
    formatted_string2label = {'error': 0, 'acceptable': 1}
    json_list_labels = get_label_display_names(formatted_string2label)
    # random.shuffle(json_list)
    save_json_lines(options.output_jsonl_file, json_list)
    save_json_lines(options.output_label_display_names_jsonl_file, json_list_labels)


if __name__ == "__main__":
    main()
