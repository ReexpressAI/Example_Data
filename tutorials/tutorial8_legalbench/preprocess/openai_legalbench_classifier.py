# Copyright © 2024 Reexpress AI, Inc.
# See the Reexpress AI tutorial for usage instructions.

"""
This serves as a simple illustration of how GPT-4 can be composed with the on-device Reexpress models. In this way,
we can readily add robust uncertainty quantification to GPT-4. This example also illustrates how we can leverage
third-party embeddings to enable estimates over documents exceeding the current 512 token limit of the on-device
models.

This is intended as a simple example for the LegalBench Tutorial, but this basic setup can be used for any
classification task. Modify `get_document_attributes()` as applicable for your task (e.g., if the expected output is
not binary Yes/No).
"""

import argparse
import json
import codecs
import time
import numpy as np
from openai import OpenAI
import glob
import os


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
            assert 'answer' in json_obj
            if json_obj['answer'] == "Yes":
                json_obj['label'] = 1
            else:
                json_obj['label'] = 0
            json_list.append(json_obj)
    return json_list


def get_document_attributes(client, document_string):

    system_prompt = f"You are a helpful assistant."
    user_prompt = f"Please answer the question. {document_string}"
    user_string = "demo1classification"
    completion = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",
             "content": user_prompt}
        ],
        max_tokens=1,
        logprobs=True,
        top_logprobs=3,
        temperature=0.0,
        user=user_string,
        seed=0
    )

    reply_text = completion.choices[0].message.content

    """
    # Here, we only generate a single token, but for reference, the topk output logits can be collected as 
    # [truncated] paths though the search lattice as follows:
    output_paths_prob = {}
    for path in range(5):
        output_paths_prob[path] = []
    """
    yes_prob = 0.0
    no_prob = 0.0
    ood_prob = 0.0
    for position_id, completion_token_position_value in enumerate(completion.choices[0].logprobs.content):
        for top_token_k, top_token in enumerate(completion_token_position_value.top_logprobs):
            token_prob = np.exp(top_token.logprob)
            if position_id == 0:
                if top_token_k < 2:
                    if top_token.token.lower() == "Yes".lower():
                        yes_prob = token_prob
                    elif top_token.token.lower() == "No".lower():
                        no_prob = token_prob
                    else:
                        ood_prob = token_prob
            # output_paths_prob[top_token_k].append(token_prob) # see comment above

    attributes = []
    attributes.append(ood_prob)
    attributes.append(yes_prob)
    attributes.append(no_prob)

    document_text = f"{user_prompt} {reply_text.strip()}"

    embedding_response = client.embeddings.create(
        model="text-embedding-3-large",
        input=document_text,
        encoding_format="float",
        user=user_string
    )

    full_embedding = embedding_response.data[0].embedding
    document_attributes = full_embedding[0:21]
    hidden_avg_split = np.split(np.array(full_embedding), indices_or_sections=8, axis=0)
    for hidden_group in hidden_avg_split:
        document_attributes.append( np.mean(hidden_group, 0) )

    document_attributes.extend(attributes)

    return reply_text, document_attributes, completion.usage.completion_tokens, completion.usage.prompt_tokens, \
        full_embedding, document_text, (document_attributes[-2] == 0 and document_attributes[-1] == 0)


def get_document_attributes_with_retry(client, document_string):  # -> Throws
    """
    For this example, we simply make two attempts to call the models and then exit.
    """
    try:
        reply_text, document_attributes, completion_tokens, prompt_tokens, embedding, document_text, is_ood = \
            get_document_attributes(client, document_string)
        return reply_text, document_attributes, completion_tokens, prompt_tokens, embedding, document_text, is_ood
    except:
        print("ERROR: Unable to retrieve model output")
        print(f"Waiting")
        time.sleep(1.0)
        print("Second attempt")
        reply_text, document_attributes, completion_tokens, prompt_tokens, embedding, document_text, is_ood = \
            get_document_attributes(client, document_string)
        return reply_text, document_attributes, completion_tokens, prompt_tokens, embedding, document_text, is_ood


def construct_minimal_json_object(id_string, label, document, attributes, prompt="", info="", group=""):
    """
    Optional fields that are empty strings are dropped. In this case, attributes are expected to be present for every
    document; however, an empty list is not included in the JSON.
    """
    # required properties
    json_obj = {"id": id_string, "label": label, "document": document}
    # optional properties
    if len(attributes) > 0:
        json_obj["attributes"] = attributes
    if len(prompt) > 0:
        json_obj["prompt"] = prompt
    if len(info) > 0:
        json_obj["info"] = info
    if len(group) > 0:
        json_obj["group"] = group

    return json_obj


def save_by_appending_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "a", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenAI GPT classification script, generating "
                                                 "attributes for input into Reexpress (macOS application).")

    parser.add_argument("--seed", type=int, default=0, help="seed")

    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing the preprocessed LegalBench data. See the Tutorial.")
    parser.add_argument(
        "--output_jsonl_dir", default="",
        help="Output directory for the data saved in the JSON lines format for input into Reexpress. See "
             "the Tutorial for the datasplits used in the example.")
    parser.add_argument(
        "--archive_output_jsonl_dir", default="",
        help="Directory saving additional properties and untruncated text and "
             "embeddings for post-hoc analysis/reference.")

    args = parser.parse_args()

    kAPI_wait_time = 0.02
    kMaxPythonChars = 4800

    client = OpenAI()

    filenames = glob.glob(f"{args.input_dir}/*.jsonl")
    filenames.sort()
    start_time = time.time()
    total_completion_tokens = 0
    total_prompt_tokens = 0
    doc_index = 0
    # total_errors = 0  # in this version we exit, if two attempts fail
    processed_files = 0
    total_files = len(filenames)
    ood_count = 0  # documents without the expected Yes | No reply
    for input_filename in filenames:
        processed_files += 1
        local_doc_index = 0
        print(f"Currently processing: {input_filename}")
        input_data_json_list = get_data(input_filename)

        for one_document_json in input_data_json_list:
            time.sleep(kAPI_wait_time)

            group_field_text = "ERROR"
            info_field_text = ""
            document_text = f"{one_document_json['text_with_prompt']}"
            document_attributes = []
            embedding = []
            id_string = one_document_json["dataset_name"] + "_" + one_document_json["index"]
            try:
                reply_text, document_attributes, completion_tokens, prompt_tokens, embedding, document_text, is_ood = \
                    get_document_attributes_with_retry(client, document_text)
                if is_ood:
                    ood_count += 1
                length_meta_info = f"{completion_tokens},{prompt_tokens}"

                total_completion_tokens += completion_tokens
                total_prompt_tokens += prompt_tokens
                group_field_text = length_meta_info
            except:
                print(f"ERROR: Unable to retrieve model output after two attempts")
                print(f"Exiting without saving document '{id_string}'")
                exit()
                # time.sleep(1.0)
                # total_errors += 1

            # if "group" in one_document_json:
            #     group_field_text = one_document_json["group"]
            if "info" in one_document_json:
                info_field_text = one_document_json["info"]

            prompt_text = ""
            json_obj = construct_minimal_json_object(id_string, one_document_json["label"],
                                                     document_text[0:kMaxPythonChars].strip(),
                                                     document_attributes, prompt=prompt_text,
                                                     info=info_field_text, group=group_field_text)

            save_by_appending_json_lines(os.path.join(args.output_jsonl_dir,
                                                 f'{one_document_json["dataset_name"]}.attributes.jsonl'), [json_obj])
            archive_json_obj = json_obj
            archive_json_obj["document_untruncated"] = document_text
            archive_json_obj["embedding"] = embedding
            # Also save the full embedding and untruncated text if needed in the future:
            save_by_appending_json_lines(os.path.join(args.archive_output_jsonl_dir,
                                                 f'{one_document_json["dataset_name"]}.archive.jsonl'),
                                         [archive_json_obj])

            local_doc_index += 1
            doc_index += 1
            print(f"Saved document {local_doc_index} of {len(input_data_json_list)} "
                  f"({processed_files} of {total_files} files. Total running documents: {doc_index})")

            if doc_index % 100 == 0:
                elapsed_seconds = time.time() - start_time
                total_tokens = total_completion_tokens + total_prompt_tokens
                print(f"Total tokens: {total_tokens}; total documents: {doc_index}")
                print(f"Elapsed seconds: {elapsed_seconds}")
                print(f"Tokens per second: {float(total_tokens) / elapsed_seconds}")
                print(f"Documents per second: {float(doc_index) / elapsed_seconds}")
                print("-"*60)
                print("")

    elapsed_seconds = time.time() - start_time
    print(f"++COMPLETE++")
    total_tokens = total_completion_tokens + total_prompt_tokens
    print(f"Total tokens: {total_tokens}; total documents: {doc_index}")
    print(f"Elapsed seconds: {elapsed_seconds}")
    print(f"Tokens per second: {float(total_tokens) / elapsed_seconds}")
    print(f"Documents per second: {float(doc_index) / elapsed_seconds}")
    print(f"Total ood count: {ood_count}")
    # print(f"Total errors: {total_errors}")