# Tutorial 6: Combine Reexpress with Mixtral-8x7B via Apple's MLX framework

### Video overview: [Here](https://youtu.be/Brm_36YRG_8)

[Reexpress](https://re.express/) can be used to add uncertainty quantification, search, and auto-visualization capabilities to essentially any generative AI language model. Tutorial 2 showed how this works using a toy example with the pre-computed output logits from the Open LLM Leaderboard. That basic setup applies to any on-premise, cloud, or local language model: Simply add the output logits as `attributes` to the document JSON lines file.

Here we provide a concrete example using the `Mixtral-8x7B-Instruct-v0.1` model, which you can run directly on your Mac using Apple's MLX framework. This model operates at the GPT-3.5 level for generative tasks, so when combined with Reexpress, many data science and enterprise language modeling tasks can be handled directly on-device. The future of personalizable, reliable AI has arrived.

In this tutorial, we will stick to the familiar task of binary sentiment classification. We will use the data from Tutorial 1. (Keep in mind that the on-device models in Reexpress are already quite strong. The on-device models are already sufficient for typical sentiment classification tasks; we stick to this task for illustrative purposes.)

For this tutorial, we will use a Mac Studio with an M2 Ultra 76-core GPU and 128 GB of unified memory, running macOS Sonoma and Reexpress one. (The basic setup here applies to any LLM that you run via MLX, including smaller models that can run on a less powerful Mac.) 


## Overview

The task is to predict whether the sentiment of a movie review is negative (class 0) or positive (class 1).

For the generative Mixtral model, we will construct a prompt and re-ask verification question as follows:

> "[INST] How can I help you?\nPlease classify the sentiment of the following review. Review: {document_string} Question: Does the previous document have a positive sentiment? Yes or No? [/INST]"

We will then take a simple transform of the final hidden layer[^1] and the output logits as the attributes that are then uploaded to Reexpress. This tutorial will proceed as follows:

1. We will install MLX; download the Mixtral weights; and quantize the model.
2. We will batch predict over the Training, Calibration, and Eval sets using our Python preprocessing code.
3. Import the JSON lines files into Reexpress and click train and predict. That's it! No additional code is needed to derive uncertainty estimates!
4. We also provide a streamlit interface for live interacting with Mixtral, including generating attributes for input into Reexpress for one-off predictions.

## 1. Install MLX and Mixtral-8x7B and quantize the model

Create a Conda environment and install MLX. (If you do not have Conda installed, see [https://conda.io/projects/conda/en/latest/user-guide/getting-started.html](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html). (This is not strictly required, but just simplifies managing your Python packages. Once installed, you can just run the following code as-is, without affecting your other Python environments.)

In the Mac Terminal application, create a Conda environment with Python 3.10.9, and install the base requirements:

```
conda create -n "mixtral" python=3.10.9
conda activate mixtral
pip install -r mixtral/requirements.txt
```

> [!NOTE]
> MLX was recently released. The code in this tutorial was originally created with version 0.0.10 and has most recently been tested with version 0.0.11. For convenience, we have included a copy of the `mixtral` directory from [https://github.com/ml-explore](https://github.com/ml-explore) as it existed when this tutorial was created. Also in that directory is the new Reexpress support code to run this tutorial.

Download the `Mixtral-8x7B-Instruct-v0.1` weights from Hugging Face via git-lfs as described in [mixtral/README.md](mixtral/README.md), which was taken from [https://github.com/ml-explore/mlx-examples/tree/main/llms/mixtral](https://github.com/ml-explore/mlx-examples/tree/main/llms/mixtral). This will create a directory `Mixtral-8x7B-Instruct-v0.1` containing the raw weights (about 195 GB).

Next, generate a quantized model:

```
conda activate mixtral

cd mixtral

export MIXTRAL_MODEL=Mixtral-8x7B-Instruct-v0.1

python convert.py \
--torch-path $MIXTRAL_MODEL/ \
--mlx-path mlx_model_quantized \
-q
```

> [!IMPORTANT]
> We will assume you have saved the quantized weights in a directory named `mlx_model_quantized` (about 29 GB). The streamlit app, in particular, assumes this naming scheme in `demo.py`.


## 2. Batch predict using Mixtral

Predict over the sentiment data provided in this repo: [/data/sentiment.zip](/data/sentiment.zip). For this demo, we will use `validation_set.jsonl.attributes.jsonl` (rather than `calibration_set.jsonl`) as the Calibration set to keep the data set size small. The following script will take several hours to run inference with Mixtral on a Mac Studio; consider running it overnight. Change the input and output directories as applicable.

```
conda activate mixtral

cd mixtral

export MIXTRAL_MODEL=Mixtral-8x7B-Instruct-v0.1

INPUT_DATA_DIR="sentiment"
OUTPUT_DATA_DIR="mixtral_sentiment_demo"
mkdir -p ${OUTPUT_DATA_DIR}

# Here we use --drop_the_prompt_in_json_output since we will just add a default prompt when creating the Reexpress project.

for INPUT_FILE in "eval_set.jsonl" "training_set.jsonl" "validation_set.jsonl"; do
python mixtral_batch_classification.py \
--model-path mlx_model_quantized \
--input_filename ${INPUT_DATA_DIR}/${INPUT_FILE} \
--output_jsonl_file ${OUTPUT_DATA_DIR}/${INPUT_FILE}.attributes.jsonl \
--prompt_text="Please classify the sentiment of the following review. Review:" \
--trailing_instruction="Question: Does the previous document have a positive sentiment?" \
--drop_the_prompt_in_json_output
done
```

**Timing**: More concretely, prediction over `eval_set.jsonl` took around 43 minutes (2582.2 seconds) for 488 documents (130,621 tokens), or roughly 50.6 tokens per second. That's actually pretty fast given the model! For context, the Reexpress **Fast I** model (3.2 billion parameters) runs inference at roughly 3,400 tokens per second (with a 3000 document support set) and the Reexpress **FastestDraft I** model (640 million parameter) runs at roughly 9,400 tokens per second. Those estimates include computationally intensive dense matching (for uncertainty) and feature importance steps. The take away is that it is quite reasonable to run both Mixtral and the Reexpress models together entirely on device for many tasks.

## 3. Create a project in Reexpress

Create a new project in Reexpress. We use the **FastestDraft I** model in the YouTube video. Use the following default prompt: 

> Please classify the sentiment of the following review.

Next, import `training_set.jsonl.attributes.jsonl`, `validation_set.jsonl.attributes.jsonl`, `eval_set.jsonl.attributes.jsonl` created above as the Training set, Calibration set, and an Eval set, respectively. Optionally, also import `label_display_names.jsonl` (provided in sentiment.zip).

Click **Learn**->**Train**->**Batch** and choose 200 epochs for `Model training` and `Model compression`. Scroll down and select the Eval set to run post-training inference. And then start!


## 4. Live prediction and interaction with Mixtral on your Mac

Once the model is trained in Reexpress, you can always use `mixtral_batch_classification.py` to predict over new data. Just import the output into Reexpress.

Alternatively, we provide a streamlit app for interacting with Mixtral in your browser for one-off predictions. 

First, [install streamlit](https://docs.streamlit.io/get-started/installation).

```
conda activate mixtral

pip install streamlit
```

This script has most recently been tested with streamlit version 1.30.0.

Once installed, you just need to run the command `streamlit run demo.py`:

```
conda activate mixtral

cd mixtral

streamlit run demo.py
```

This will make the UI accessible from localhost and a corresponding URL on your local network. To adjust the streamlit settings, please refer to the documentation at [https://streamlit.io/](https://streamlit.io/). The file [mixtral/.streamlit/config.toml](mixtral/.streamlit/config.toml) contains config options.


Choose `Chat` mode for a standard chat interface. Use `Re-ask` mode to generate Reexpression attributes which you can then copy-and-paste into Reexpress. (Remember that if you train the model with 'attributes' from another model/source, you will want to include such attributes at test time, as well.) See the YouTube video above for an example.

> [!TIP]
> To stop streamlit, got to the Terminal and type `control-c`.

## Tips

Mixtral-8x7B-Instruct-v0.1 is sufficiently fast on a Mac Studio with an M2 Ultra 76-core GPU and 128 GB of unified memory for small-scale datasets (on the order of several thousand documents, which is a scale that can be processed over night, depending on the document length). For larger datasets, consider running inference over the training set on a server (or split the files across multiple Macs, since inference is embarrassingly parallel across documents) and then move on-device for inference. Only a single pass over the data with the generative AI model is needed; subsequent training and re-training is possible just via the cached attributes in the JSON lines files. (A useful implication of this is that you can run the generative AI model over larger-scale generic or publicly available datasets in the cloud, and then you can add training instances from your IP-protected data by running the model over those documents on your Mac. Subsequent inference, and the model itself, can then stay on your, or your company's, Mac(s).)

> [!NOTE]
> In this example, we saw how to turn free-form generation into classification via a simple binary re-ask prompt+trailing instruction, and the underlying task itself was binary classification. This basic setup also applies for >=3-class classification. A typical approach would be to have the generative model predict the label, and then construct the re-ask trailing instruction based on that label.

> [!TIP]
> Most importantly: Keep in mind that the on-device models built into Reexpress are already quite strong. In this tutorial, we've used the task of sentiment classification since it's straightforward for everyone to easily understand the prediction task. In practice, the on-device models are already sufficiently strong for this task and you don't need another model!

Another use-case for the Mixtral model is to generate synthetic data to bootstrap a classification task, after which running the generative model may not be necessary. We saw a toy example of this in the YouTube video for this tutorial when creating a sample movie review. We will re-visit this point in future tutorials.

[^1]: In this case, we have access to the model so we also include a simple transform of the hidden layer, rather than just the output logits (as in Tutorial 2), as attributes. Why? The Mixtral model has a much longer input context than the 512 tokens of the on-device Reexpress models, so including the hidden layer values potentially enables greater specificity for constructing the reference classes internally in Reexpress. (The uncertainty estimate would still be valid by just using the output logits, but the additional signal is a guard on the edge case of marginalizing over the content between 512 tokens and the max Mixtral length, if applicable. That is, the 'interpretability by exemplar matching' analysis functionality can be more informative for such long inputs, if applicable.) Technical aside: The raw attribute values are not directly used to calculate L2 distances. Rather, they are composed with the on-device models and take part in the training process to construct the overall model. The model will learn whether or not the provided attributes are useful signals for the given classification task. As such, provided you can consistently generate such attributes at training AND test, it is generally ok to include attributes that may not be correlated with the task and let the model determine their significance.
