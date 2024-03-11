# Preprocess the LegalBench data

LegalBench is a large, publicly available dataset for evaluating the legal reasoning capabilities of LLMs. We focus on the subset of 80 tasks structured as binary classification (with a "Yes" or "No" answer) that also have an MIT or CC BY 4.0 license. This subset of the data is available [here](https://drive.google.com/file/d/1TnSRMh3dPwJtWxZao5rfYxHZhOGyuLSx/view?usp=sharing), which is the subset of the full data available from the LegalBench [repo](https://github.com/HazyResearch/legalbench). Note that we have formatted the documents using the base prompt from the repo.

The end result of the following will be to produce JSON lines files that contain a vector of 32 elements associated with each document. If you want to skip the following, you can download the data formatted for import into Reexpress, split into a Training, Calibration, and Eval set [here](https://drive.google.com/file/d/1_GofjFb8g9sO9iDhgYvh_RS2eorsS_km/view?usp=sharing). We also provide all of the data by task [here](https://drive.google.com/file/d/1Hs3W_pBFKcfUxV_eLpQrVtChF6f0yshk/view?usp=sharing), as a convenience if you want to examine the behavior of Reexpress on other splits, including distribution shifts by task.

To prepare the data for import into Reexpress, we will:
- Run inference using `gpt-4-0125-preview` and `text-embedding-3-large`.
- Randomly split the data into Training, Calibration, and Eval datasplits.

## Run GPT-4 and the embedding model

Install the OpenAI Python library:

```
pip install openai
```

Run inference, changing the input/output directories as desired:

```
INPUT_DATA_DIR="legalbench_raw_subset"
OUTPUT_DATA_DIR="legalbench_processed/attributes_openai_gpt-4-0125-preview"
ARCHIVE_OUTPUT_DATA_DIR="legalbench_processed/attributes_openai_gpt-4-0125-preview_archive"
mkdir -p ${OUTPUT_DATA_DIR}
mkdir -p ${ARCHIVE_OUTPUT_DATA_DIR}

python openai_legalbench_classifier.py \
--input_dir ${INPUT_DATA_DIR} \
--output_jsonl_dir ${OUTPUT_DATA_DIR} \
--archive_output_jsonl_dir ${ARCHIVE_OUTPUT_DATA_DIR}
```

> [!NOTE]
> The directory `${ARCHIVE_OUTPUT_DATA_DIR}` saves the full text and uncompressed embeddings for future reference or analysis.

## Datasplits

Next, separate the data into Training, Calibration, and Eval splits:

```
INPUT_DATA_DIR="legalbench_processed/attributes_openai_gpt-4-0125-preview"
OUTPUT_DATA_DIR="legalbench_processed/attributes_openai_gpt-4-0125-preview_all_split_within_tasks"

mkdir -p ${OUTPUT_DATA_DIR}

python -u openai_legalbench_split_data_within_task.py \
--input_dir ${INPUT_DATA_DIR} \
--output_jsonl_dir ${OUTPUT_DATA_DIR}
```

This data is available via the link above, including a label display names file. Alternatively, if you want to examine the behavior of Reexpress on distribution shifts (as in the previous Tutorial), you can use `openai_legalbench_split_data_by_task.py`, which splits the data by task. 

## Next

And that's it! You can then import the processed output into Reexpress, which will then do all of the analysis heavy-lifting for you without needing to write any additional analysis code!

> [!TIP]
> Remember, if you then subsequently want to make a prediction over a new document (or run a semantic search), you will need to process that document with `openai_legalbench_classifier.py` to generate the attributes to import into Reexpress.
