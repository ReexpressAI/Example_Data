# Preprocess the SAPLMA data

For this tutorial, we will use a Mac Studio with an M2 Ultra 76-core GPU and 128 GB of unified memory, running macOS Sonoma and Reexpress one. (If you have a less powerful Mac and want to follow the tutorial, you can skip this preprocessing step and download the preprocessed data at [../data/saplma_true_false_data_with_mixtral_instruct_v0.1.zip](../data/saplma_true_false_data_with_mixtral_instruct_v0.1.zip).) 

Download the original data from [http://azariaa.com/Content/Datasets/true-false-dataset.zip](http://azariaa.com/Content/Datasets/true-false-dataset.zip), which is a link present in the original paper, ["The Internal State of an LLM Knows When It’s Lying"](https://arxiv.org/pdf/2304.13734v2.pdf).

We assume that you have already followed [Tutorial 6](/tutorials/tutorial6_mlx/README.md), and as such, we assume that you have created a conda environment called `mixtral` and downloaded and quantized the `Mixtral-8x7B-Instruct-v0.1` model.


First, we convert the original CSV-formatted data into the Reexpress JSON lines format. Change `INPUT_DATA_DIR` to the unzipped data directory downloaded above.

```
conda activate mixtral

INPUT_DATA_DIR="publicDataset"
OUTPUT_DATA_DIR="publicDataset/processed"
mkdir -p ${OUTPUT_DATA_DIR}

for INPUT_FILE in "animals_true_false.csv" "companies_true_false.csv" "facts_true_false.csv" "inventions_true_false.csv" "cities_true_false.csv" "elements_true_false.csv" "generated_true_false.csv"; do
echo "Processing ${INPUT_FILE}"
python saplma_dataset.py \
--input_data_path ${INPUT_DATA_DIR} \
--input_data_file ${INPUT_FILE} \
--output_jsonl_file ${OUTPUT_DATA_DIR}/${INPUT_FILE}.jsonl \
--output_label_display_names_jsonl_file ${OUTPUT_DATA_DIR}/label_dislay_names.jsonl
done

```

Next, we will run a single inference pass over the data using the `Mixtral-8x7B-Instruct-v0.1` model, caching the results in the attributes field in the new JSON lines output file. To do this, we will use the model and script from [Tutorial 6](/tutorials/tutorial6_mlx/README.md), but with a different prompt and trailing instruction. Change the directories, as needed.


```
conda activate mixtral

cd /tutorials/tutorial6_mlx/mixtral


INPUT_DATA_DIR="publicDataset/processed"
OUTPUT_DATA_DIR="publicDataset/processed/attributes"
mkdir -p ${OUTPUT_DATA_DIR}

# Here we use --drop_the_prompt_in_json_output since we will just add a default prompt when creating the Reexpress project.

for INPUT_FILE in "animals_true_false.csv" "companies_true_false.csv" "facts_true_false.csv" "inventions_true_false.csv" "cities_true_false.csv" "elements_true_false.csv" "generated_true_false.csv"; do
echo "Processing ${INPUT_FILE}"
python mixtral_batch_classification.py \
--model-path mlx_model_quantized \
--input_filename ${INPUT_DATA_DIR}/${INPUT_FILE}.jsonl \
--output_jsonl_file ${OUTPUT_DATA_DIR}/${INPUT_FILE}.attributes.jsonl \
--prompt_text="Given the following document, please answer the question. Document:" \
--trailing_instruction="Question: Does the previous document contain correct information and is it well-written?" \
--drop_the_prompt_in_json_output
done
```

## Tips

Remember, if you then subsequently want to make a prediction over a new document, you will need to process that document with `mixtral_batch_classification.py` (or via the Streamlit app `streamlit run demo.py` from [Tutorial 6](/tutorials/tutorial6_mlx/README.md)) to generate the attributes to import into Reexpress.

Feel free to modify the Mixtral prompt and trailing instruction for your task/data. The important part is being consistent across training and test. For this tutorial, we used the following prompt:

>Given the following document, please answer the question. Document:

And trailing instruction:

>Question: Does the previous document contain correct information and is it well-written?
