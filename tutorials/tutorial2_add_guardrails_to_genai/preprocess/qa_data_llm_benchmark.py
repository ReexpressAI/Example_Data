"""
This constructs example data that demonstrates ensembling the on-device models with an additional model; that is
ADDING UNCERTAINTY QUANTIFICATION TO ESSENTIALLY ANY LANGUAGE MODEL.
By default, this uses Mistral 7B v0.1
(https://huggingface.co/datasets/open-llm-leaderboard/details_mistralai__Mistral-7B-v0.1),
"a 7-billion-parameter language model engineered for superior performance and efficiency"
(https://arxiv.org/abs/2310.06825). Some archived model output on the Open LLM Leaderboard may have different names
for the dictionary keys; the script will need to be modified accordingly.

The takeaway: You can add the output logits of your existing model to add uncertainty quantification, search, and the
other analysis capabilities of the Reexpress application.

Here we simply pull data from the Open LLM Leaderboard (https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard),
which contains the output logits from the model. We add the output logits to the JSON lines files as attributes. The
outputs across datasets are on rather different scales; for the purposes of this tutorial, we perform a simple,
naive normalization. For reference, we also add 1-hot indicators of the predictions to show what we mean by
turning your model's output into a categorial idicator: This can be useful if your
LLM does not expose the output logits, which is currently the case with some cloud APIs.

We re-split this eval data into training, calibration, and eval datasplits from the following datasets:

HellaSwag https://arxiv.org/abs/1905.07830
AI2 Reasoning Challenge https://arxiv.org/pdf/1803.05457.pdf
MMLU https://arxiv.org/abs/2009.03300

This is only a small example for simple demonstration purposes. As we note in the tutorial video, by design, the
training and calibration datasplits contain some documents that are distribution-shifted relative to the eval split.
Obviously, for a full held-out Open LLM eval, use the full training and calibration sets of the original datasets,
which will require running the original third-party LLM, and use the standard test sets for eval.

Note that unique UUIDs for each document will be created each time this script is run.

For the video tutorial, this script was run as follows:

python -u qa_data_split_test.py \
--model_name "open-llm-leaderboard/details_mistralai__Mistral-7B-v0.1" \
--config_split_date_key "2023_09_27T15_30_59.039834" \
--output_train_jsonl_file ${OUTPUT}/"train.jsonl" \
--output_calibration_jsonl_file ${OUTPUT}/"calibration.jsonl" \
--output_eval_jsonl_file ${OUTPUT}/"eval.jsonl"

#Total MMLU configs: 57
#{'train': 16, 'calibration': 10, 'eval': 31}

"""

import argparse
import codecs
import json
import uuid
from datasets import load_dataset
import torch


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def get_mmlu_json_lines_for_config(model_name, mmlu_config, config_split_date_key):
    data = load_dataset(model_name,
                        mmlu_config,
                        split=config_split_date_key)
    json_list = []
    doc_i = 0
    for query, predictions, choices, gold_int in zip(data['query'], data['predictions'], data['choices'], data['gold']):
        document = ["Question:", query[0:-len("Answer:")]]
        predictions_normed = [float(x) for x in list(torch.nn.functional.normalize(
            torch.tensor(predictions).unsqueeze(0)).squeeze().numpy())]
        one_hot_indicator = []
        prediction_int = int(torch.argmax(torch.tensor(predictions)).item())
        for choice_i in range(4):
            if prediction_int == choice_i:
                one_hot_indicator.append(1.0)
            else:
                one_hot_indicator.append(0.0)
        predictions_normed.extend(one_hot_indicator)
        json_obj = {"id": str(uuid.uuid4()), "label": gold_int,
                          "document": " ".join(document),
                          "info": mmlu_config, "group": f"{doc_i}",
                          "attributes": predictions_normed}
        json_list.append(json_obj)
        doc_i += 1
    return json_list


# Split MMLU eval in half by category, splitting into a new eval and redistributing the rest among training and calibration
def get_mmlu_json_lines_from_sample_of_available_configs(model_name, config_split_date_key):
    # combined config is as follows: 'harness_hendrycksTest_5'
    mmlu_all = ['harness_hendrycksTest_abstract_algebra_5', 'harness_hendrycksTest_anatomy_5',
                'harness_hendrycksTest_astronomy_5',
     'harness_hendrycksTest_business_ethics_5', 'harness_hendrycksTest_clinical_knowledge_5',
     'harness_hendrycksTest_college_biology_5', 'harness_hendrycksTest_college_chemistry_5',
     'harness_hendrycksTest_college_computer_science_5', 'harness_hendrycksTest_college_mathematics_5',
     'harness_hendrycksTest_college_medicine_5', 'harness_hendrycksTest_college_physics_5',
     'harness_hendrycksTest_computer_security_5', 'harness_hendrycksTest_conceptual_physics_5',
     'harness_hendrycksTest_econometrics_5', 'harness_hendrycksTest_electrical_engineering_5',
     'harness_hendrycksTest_elementary_mathematics_5', 'harness_hendrycksTest_formal_logic_5',
     'harness_hendrycksTest_global_facts_5', 'harness_hendrycksTest_high_school_biology_5',
     'harness_hendrycksTest_high_school_chemistry_5', 'harness_hendrycksTest_high_school_computer_science_5',
     'harness_hendrycksTest_high_school_european_history_5', 'harness_hendrycksTest_high_school_geography_5',
     'harness_hendrycksTest_high_school_government_and_politics_5',
     'harness_hendrycksTest_high_school_macroeconomics_5', 'harness_hendrycksTest_high_school_mathematics_5',
     'harness_hendrycksTest_high_school_microeconomics_5', 'harness_hendrycksTest_high_school_physics_5',
     'harness_hendrycksTest_high_school_psychology_5', 'harness_hendrycksTest_high_school_statistics_5',
     'harness_hendrycksTest_high_school_us_history_5', 'harness_hendrycksTest_high_school_world_history_5',
     'harness_hendrycksTest_human_aging_5', 'harness_hendrycksTest_human_sexuality_5',
     'harness_hendrycksTest_international_law_5', 'harness_hendrycksTest_jurisprudence_5',
     'harness_hendrycksTest_logical_fallacies_5', 'harness_hendrycksTest_machine_learning_5',
     'harness_hendrycksTest_management_5', 'harness_hendrycksTest_marketing_5',
     'harness_hendrycksTest_medical_genetics_5', 'harness_hendrycksTest_miscellaneous_5',
     'harness_hendrycksTest_moral_disputes_5', 'harness_hendrycksTest_moral_scenarios_5',
     'harness_hendrycksTest_nutrition_5', 'harness_hendrycksTest_philosophy_5', 'harness_hendrycksTest_prehistory_5',
     'harness_hendrycksTest_professional_accounting_5', 'harness_hendrycksTest_professional_law_5',
     'harness_hendrycksTest_professional_medicine_5', 'harness_hendrycksTest_professional_psychology_5',
     'harness_hendrycksTest_public_relations_5', 'harness_hendrycksTest_security_studies_5',
     'harness_hendrycksTest_sociology_5', 'harness_hendrycksTest_us_foreign_policy_5',
     'harness_hendrycksTest_virology_5', 'harness_hendrycksTest_world_religions_5']

    print(f"Total MMLU configs: {len(mmlu_all)}")

    indexes = torch.randint(0, 4, (len(mmlu_all),))
    mmlu_config_split_dict = {"train": 0, "calibration": 0, "eval": 0}
    mmlu_json_lines_split_dict = {"train": [], "calibration": [], "eval": []}
    for i, mmlu_config in enumerate(mmlu_all):
        if indexes[i].item() == 0:
            mmlu_config_split_dict["train"] += 1
            mmlu_json_lines_split_dict["train"].extend(get_mmlu_json_lines_for_config(
                model_name, mmlu_config, config_split_date_key))
        elif indexes[i].item() == 1:
            mmlu_config_split_dict["calibration"] += 1
            mmlu_json_lines_split_dict["calibration"].extend(get_mmlu_json_lines_for_config(
                model_name, mmlu_config, config_split_date_key))
        else:
            mmlu_config_split_dict["eval"] += 1
            mmlu_json_lines_split_dict["eval"].extend(get_mmlu_json_lines_for_config(
                model_name, mmlu_config, config_split_date_key))
    print(mmlu_config_split_dict)
    return mmlu_json_lines_split_dict


def process_arc(options):
    data = load_dataset(options.model_name,
                        "harness_arc_challenge_25",
                        split=options.config_split_date_key)

    json_list = []
    doc_i = 0
    for query, predictions, choices, gold_int in zip(data['query'], data['predictions'], data['choices'], data['gold']):
        document = [query[0:-len("Answer:")]]
        possible_letters = ["A.", "B.", "C.", "D.", "E.", "F."]
        one_hot_indicator = []
        prediction_int = int(torch.argmax(torch.tensor(predictions)).item())
        for choice_i, choice in enumerate(choices):
            document.append(f"{possible_letters[choice_i]} {choice}")
            if prediction_int == choice_i:
                one_hot_indicator.append(1.0)
            else:
                one_hot_indicator.append(0.0)
        predictions_normed = [float(x) for x in list(torch.nn.functional.normalize(torch.tensor(predictions).unsqueeze(0)).squeeze().numpy())]
        predictions_normed.extend(one_hot_indicator)
        json_obj = {"id": str(uuid.uuid4()), "label": gold_int,
                          "document": " ".join(document),
                          "info": "harness_arc_challenge_25", "group": f"{doc_i}",
                          "attributes": predictions_normed}
        json_list.append(json_obj)
        doc_i += 1
    return json_list


def process_swag(options, json_list):
    data = load_dataset(options.model_name,
                        "harness_hellaswag_10",
                        split=options.config_split_date_key)
    doc_i = 0
    for query, predictions, choices, gold_int in zip(data['query'], data['predictions'], data['choices'], data['gold']):
        document = ["Question: Please choose the most likely completion of the following scenario:", query]
        possible_letters = ["A.", "B.", "C.", "D.", "E.", "F.", "G."]
        one_hot_indicator = []
        prediction_int = int(torch.argmax(torch.tensor(predictions)).item())
        for choice_i, choice in enumerate(choices):
            document.append(f"{possible_letters[choice_i]} {choice}")
            if prediction_int == choice_i:
                one_hot_indicator.append(1.0)
            else:
                one_hot_indicator.append(0.0)
        predictions_normed = [float(x) for x in list(torch.nn.functional.normalize(torch.tensor(predictions).unsqueeze(0)).squeeze().numpy())]
        predictions_normed.extend(one_hot_indicator)
        json_obj = {"id": str(uuid.uuid4()), "label": gold_int,
                          "document": " ".join(document),
                          "info": "harness_hellaswag_10", "group": f"{doc_i}",
                          "attributes": predictions_normed}
        json_list.append(json_obj)
        doc_i += 1
    return json_list


def main():
    parser = argparse.ArgumentParser(
        description="-----[Output JSON lines format.]-----")
    parser.add_argument(
        "--model_name", default="open-llm-leaderboard/details_mistralai__Mistral-7B-v0.1",
        help="Model name used on the Open LLM Leaderboard")
    parser.add_argument(
        "--config_split_date_key", default="2023_09_27T15_30_59.039834",
        help="Model name used on the Open LLM Leaderboard")
    parser.add_argument(
        "--output_train_jsonl_file", default="",
        help="JSON lines output file. Must have the ending .jsonl")
    parser.add_argument(
        "--output_calibration_jsonl_file", default="",
        help="JSON lines output file. Must have the ending .jsonl")
    parser.add_argument(
        "--output_eval_jsonl_file", default="",
        help="JSON lines output file. Must have the ending .jsonl")

    options = parser.parse_args()
    torch.manual_seed(0)

    json_list = process_arc(options)
    json_list = process_swag(options, json_list)

    indexes = torch.randint(0, 2, (len(json_list),))
    train = []
    calibration = []
    idx = 0
    for json_obj in json_list:
        is_train = indexes[idx].item() == 0
        if is_train:
            train.append(json_obj)
        else:
            calibration.append(json_obj)
        idx += 1

    mmlu_json_lines_split_dict = get_mmlu_json_lines_from_sample_of_available_configs(options.model_name,
                                                                                      options.config_split_date_key)

    train.extend(mmlu_json_lines_split_dict["train"])
    save_json_lines(options.output_train_jsonl_file, train)

    calibration.extend(mmlu_json_lines_split_dict["calibration"])
    save_json_lines(options.output_calibration_jsonl_file, calibration)

    save_json_lines(options.output_eval_jsonl_file, mmlu_json_lines_split_dict["eval"])


if __name__ == "__main__":
    main()



