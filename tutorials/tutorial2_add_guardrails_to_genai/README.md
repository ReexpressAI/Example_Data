# Tutorial 2: Adding Uncertainty Estimates to a Generative Language Model: Example Data

Example data for Reexpress one (macOS application) for the following: [Tutorial 2: Adding Uncertainty Estimates to a Generative Language Model](https://youtu.be/5HzD3NwKc-U).

The [zip archive](/data/simple_qa.zip) contains a subset of Multiple Choice Question Answering examples from the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard). We re-split this eval data into training, calibration, and eval datasplits from the following datasets:

- [HellaSwag](https://arxiv.org/abs/1905.07830)
- [AI2 Reasoning Challenge](https://arxiv.org/pdf/1803.05457.pdf)
- [MMLU](https://arxiv.org/abs/2009.03300)


To demonstrate combining the on-device model with a generative language model, we add the output [logits](https://huggingface.co/datasets/open-llm-leaderboard/details_mistralai__Mistral-7B-v0.1) of [Mistral 7B v0.1](https://arxiv.org/abs/2310.06825), "a 7-billion-parameter language model engineered for superior performance and efficiency", as attributes to each corresponding document in the input JSON lines file. For reference, the preprocessing script is [here](preprocess/qa_data_llm_benchmark.py), but that script is not particularly important for the purposes of this tutorial or otherwise; the data can just be downloaded directly from the zip archive above.

*(The outputs across datasets are on rather different scales; for the purposes of this tutorial, we perform a simple, naive normalization. For reference, we also add 1-hot indicators of the predictions to show what we mean by turning your model's output into a categorial idicator: This can be useful if your LLM does not expose the output logits, which is currently the case with some cloud APIs.)*


**The takeaway:**

> You can add the output logits of your existing model to add uncertainty quantification, search, and the other analysis capabilities of the Reexpress application.

## Getting started

Unzip the archive and create a project for 4-class classification, corresponding to each of the 4 answer choices, A., B., C., and D. Add the data to the Training set, Calibration set, and Eval set, matching the filenames, via **Data**->**Add**.

To change the label names displayed in the interface from numbers (0,1,2,3) to ("A", "B", "C", "D"), upload the file `label_display_names.jsonl` via **Data**->**Labels**.

Next, go to the **Learn** tab and start training.

That's all that's needed to start the data analysis journey, which includes dense (vector) matching, semantic searches, auto visualizations, uncertainty quantification, and more...all without writing a single line of analysis code nor messing with tricky GPU drivers nor sending your data to an external server!


*Tip: This is only a small example for simple demonstration purposes. As we note in the tutorial video, by design, the training and calibration datasplits contain some documents that are distribution-shifted relative to the eval split. Obviously, for a full held-out Open LLM eval, use the full training and calibration sets of the original datasets, which will require running the original third-party LLM, and use the standard test sets for eval.*
