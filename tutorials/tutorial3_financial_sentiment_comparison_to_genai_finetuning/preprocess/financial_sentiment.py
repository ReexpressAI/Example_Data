"""
This simple example demonstrates how much easier it is to run a full data analysis pipeline directly on your Mac using
Reexpress than to build a solution from existing models.

Here, the task is to determine the sentiment of financial news headlines, classifying them as negative, neutral,
or positive.

The overall point accuracy is similar to fine-tuning recent 7b generative models (with accuracy around 87 when using the Fast I model),
as shown in the third-party Medium post and Kaggle script linked below, but with the
indispensable additional advantages of uncertainty quantification, interpretability by example/exemplar,
and semantic search capabilities. These latter capabilities in particular are rather non-trivial to implement
from scratch.

This incorporates Apache 2.0 code linked from https://medium.com/@lucamassaron/fine-tuning-a-large-language-model-on-kaggle-notebooks-for-solving-real-world-tasks-part-3-f15228f1c2a2
 See https://www.kaggle.com/code/lucamassaron/fine-tune-mistral-v0-2-for-sentiment-analysis

The data is available here:
https://www.kaggle.com/code/lucamassaron/fine-tune-mistral-v0-2-for-sentiment-analysis/input

Note that unique UUIDs for each document will be created each time this script is run.

We use the following prompt for all documents, which is included in the JSON lines files:
Please classify the sentiment of the following financial news headline as positive, neutral, or negative, explaining your reasoning step by step.

Run this script as follows, updating the global constants INPUT_DATA and OUTPUT as applicable:

INPUT_DATA=".../all-data.csv"
OUTPUT="...sentiment_finance_public"
mkdir -p ${OUTPUT}

python -u financial_sentiment.py \
--input_data_csv ${INPUT_DATA} \
--output_train_jsonl_file ${OUTPUT}/"train.jsonl" \
--output_calibration_jsonl_file ${OUTPUT}/"calibration.jsonl" \
--output_test_jsonl_file ${OUTPUT}/"test.jsonl" \
--output_label_display_names_jsonl_file ${OUTPUT}/"sentiment_3class_labels.jsonl"

Next, just upload the data to Reexpress and train. We recommend using the Fast I (3.2 billion parameter) model.
Training for 200 epochs, otherwise using the default settings, is a good place to start.
Once trained, you then get the additional capabilities mentioned above without writing any analysis code!
"""

import argparse
import codecs
import json
import uuid
import pandas as pd
from sklearn.model_selection import train_test_split


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def get_json_list(dataset_pd_obj, string2label):
    json_list = []

    for index, row in dataset_pd_obj.iterrows():
        json_obj = {"id": str(uuid.uuid4()), "label": string2label[row["sentiment"]],
                    "document": row["text"],
                    "prompt": "Please classify the sentiment of the following financial news headline as positive, neutral, or negative, explaining your reasoning step by step.",
                    "info": f"{index}"}
        json_list.append(json_obj)
    return json_list


def process_data(filename, string2label):
    """
    Process train and test data as in
    https://www.kaggle.com/code/lucamassaron/fine-tune-mistral-v0-2-for-sentiment-analysis

    Note that there is a bug in the above comparison code for constructing the eval set, since indexes in
    train/test could be included in the sample. Instead, here we simply construct the eval set (used for calibration)
    from all remaining documents not in train and test.
    """
    df = pd.read_csv(filename,
                     names=["sentiment", "text"],
                     encoding="utf-8", encoding_errors="replace")
    X_train = []
    X_test = []
    covered_idx = []
    for sentiment in ["positive", "neutral", "negative"]:
        train, test = train_test_split(df[df.sentiment == sentiment],
                                       train_size=300,
                                       test_size=300,
                                       random_state=42)
        X_train.append(train)
        X_test.append(test)
        covered_idx.extend(list(train.index) + list(test.index))  # such a line is missing in the original Kaggle code

    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_test = pd.concat(X_test)
    eval_idx = [idx for idx in df.index if idx not in covered_idx]
    X_eval = df[df.index.isin(eval_idx)]
    X_train = X_train.reset_index(drop=True)

    json_list_train = get_json_list(X_train, string2label)
    json_list_calibration = get_json_list(X_eval, string2label)
    json_list_test = get_json_list(X_test, string2label)

    return json_list_train, json_list_calibration, json_list_test


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
        "--input_data_csv", default="all-data.csv",
        help="Data from https://www.kaggle.com/code/lucamassaron/fine-tune-mistral-v0-2-for-sentiment-analysis/input")
    parser.add_argument(
        "--output_train_jsonl_file", default="",
        help="JSON lines output file. Must have the ending .jsonl")
    parser.add_argument(
        "--output_calibration_jsonl_file", default="",
        help="JSON lines output file. Must have the ending .jsonl")
    parser.add_argument(
        "--output_test_jsonl_file", default="",
        help="JSON lines output file. Must have the ending .jsonl")
    parser.add_argument(
        "--output_label_display_names_jsonl_file", default="",
        help="JSON lines output file for label display names. Must have the ending .jsonl")

    options = parser.parse_args()

    string2label = {'positive': 2, 'neutral': 1, 'negative': 0}
    json_list_train, json_list_calibration, json_list_test = process_data(options.input_data_csv, string2label)
    json_list_labels = get_label_display_names(string2label)

    save_json_lines(options.output_train_jsonl_file, json_list_train)
    save_json_lines(options.output_calibration_jsonl_file, json_list_calibration)
    save_json_lines(options.output_test_jsonl_file, json_list_test)

    save_json_lines(options.output_label_display_names_jsonl_file, json_list_labels)


if __name__ == "__main__":
    main()
