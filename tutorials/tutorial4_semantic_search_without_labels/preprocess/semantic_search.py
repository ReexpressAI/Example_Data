"""
Reexpress is structured around the task of supervised document classification. However, we can also easily unlock
the semantic search capabilities on essentially any document collection. It's just a matter of leveraging
naturally occurring labels to assign contrastive labels so that you can train your custom model.
This is typically easy in practice; "Tutorial 4: Semantic Search without Labeled Documents" presents some examples.

Here, we preprocess Project Gutenberg data to demonstrate this point, using
Charles Dickens' books Great Expectations (1861) and A Tale of Two Cities (1859). As an end goal, we don't
particularly need to classify passages from these two books (after all, we have the labels, and we won't be encountering
new unlabeled documents from these two distributions, per se), but by setting up the
problem as a classification task, we can then unlock the semantic search capabilities of Reexpress.

See Tutorial 4 for additional details.

Note that unique UUIDs for each document will be created each time this script is run.

You may be able to use this for other Project Gutenberg books, but note that this script assumes
the global constants kStart_maker and kEnd_maker delimit the core text of interest, which may or may not be true for
other books.

"""

import argparse
import codecs
import json
import uuid
from os import path

# Constants
kStart_maker = "*** START OF THE PROJECT GUTENBERG EBOOK"
kEnd_maker = "*** END OF THE PROJECT GUTENBERG EBOOK"


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def get_json_list(filename_with_path, label_int, prompt):
    """
    This assumes the global constants kStart_maker and kEnd_maker delimit the core text of interest.
    """
    json_list = []
    combined_lines = []
    start_parsing_counter = 0
    with codecs.open(filename_with_path, "r", encoding="utf-8") as f:
        for line in f:
            if start_parsing_counter == 0 and kStart_maker in line:
                start_parsing_counter += 1
            elif start_parsing_counter != 0 and kEnd_maker in line:
                start_parsing_counter += 1
            elif start_parsing_counter == 1:  # skip the kStart_maker line itself
                start_parsing_counter += 1

            if start_parsing_counter == 2:
                if line.strip() == "":
                    combined_lines.append("\n")
                else:
                    combined_lines.append(line.strip())
    combined_lines_string = " ".join(combined_lines)
    index = 0
    for line in combined_lines_string.split("\n"):
        if line.strip() != "":
            # The leading zeros in line_index_string allow us to sort the documents in their original order
            # by going to Explore->Select->Sorting and choosing the 'info' field, if desired.
            line_index_string = "{:06d}".format(index)
            json_obj = {"id": str(uuid.uuid4()), "label": label_int,
                        "document": line.strip(),
                        "prompt": prompt.strip(),
                        "info": line_index_string}
            json_list.append(json_obj)
            index += 1
    return json_list


def main():
    parser = argparse.ArgumentParser(
        description="-----[Construct output JSON lines formatted files for input to Reexpress.]-----")
    parser.add_argument(
        "--input_data_txt", default="input_data.txt",
        help="A Project Gutenberg eBook .txt file.")
    parser.add_argument(
        "--book_label", default="Great Expectations",
        help="String label used in the output filename to differentiate books. Consider just using the book's title.")
    parser.add_argument(
        "--prompt",
        default="Please classify the topic of the following book excerpt, explaining your reasoning step by step.",
        help="Prompt assigned to each document.")
    parser.add_argument(
        "--label_int", default=0, type=int,
        help="Label assigned to each document.")
    parser.add_argument('--split_into_training_and_calibration_sets', action='store_true',
                        help="If provided, the input file is split 50/50 into a Training set and a Calibration set.")
    parser.add_argument(
        "--output_directory", default="",
        help="JSON lines output files (with the required file extension .jsonl) will be placed in this directory.")

    options = parser.parse_args()
    json_list = get_json_list(options.input_data_txt, options.label_int, options.prompt)
    if options.split_into_training_and_calibration_sets:
        save_json_lines(path.join(options.output_directory, f"{options.book_label}_training.jsonl"),
                        json_list[0:len(json_list)//2])
        save_json_lines(path.join(options.output_directory, f"{options.book_label}_calibration.jsonl"),
                        json_list[len(json_list)//2:])
    else:
        save_json_lines(path.join(options.output_directory, f"{options.book_label}_complete.jsonl"),
                        json_list)


if __name__ == "__main__":
    main()
