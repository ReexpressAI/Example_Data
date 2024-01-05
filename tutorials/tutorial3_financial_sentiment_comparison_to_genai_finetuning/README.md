# Tutorial 3: Comparing Reexpress to Fine-tuning a Generative AI Model for Classification

**TL;DR:** [Reexpress](https://re.express/) *makes it really easy (and inexpensive) to fine-tune a large language model (LLM) for typical document classification tasks. All of the processing happens on your Mac and you also get the indispensable additional advantages of uncertainty quantification, interpretability by example/exemplar, and semantic search capabilities.* 

With real-world enterprise and data science use cases, relying on prompts alone to update a generative model for a classification task is often not sufficient. It is typically recommended to fine-tune the large language model (LLM). For real-world use cases, we already have some amount of labeled data to evaluate the LLM and assign uncertainty estimates, so we can use some subset of such data to fine-tune the model.

However, what is the best way to fine-tune an LLM? Typically this is a more expensive option with cloud APIs, both for initial fine-tuning and for subsequently using the model. And with open-source models, it requires rather more technical know-how than inference alone. Luckily, fine-tuning open-source LLMs is getting more straightforward. In a great recent Medium post, ["Fine-tuning a large language model on Kaggle Notebooks for solving real-world tasks — part 3"](https://medium.com/@lucamassaron/fine-tuning-a-large-language-model-on-kaggle-notebooks-for-solving-real-world-tasks-part-3-f15228f1c2a2), it is shown how to fine-tune Mistral 7b and Phi-2 using a Kaggle notebook. Here, the task is to determine the sentiment of financial news headlines, classifying them as negative, neutral, or positive.

But can we make it even easier, with a no-code approach? And importantly, what do we do after fine-tuning? Fine-tuning is actually only the initial step. Completing the full data analysis pipeline after fine-tuning is quite complex. We need uncertainty quantification and interpretability by example/exemplar. What if we want to run semantic searches to further examine the data? These things are quite difficult to implement.

That's where Reexpress comes in. And in fact, for a task such as sentiment classification, you do not even need another model. There is no need to ensemble with another model as we saw with Tutorial 2; the on-device models are sufficient. The end result is a dramatically simplified pipeline to get to very advanced data analysis capabilities.

It's easy to compare to the fine-tuning examples above that used Mistral 7b and Phi-2. We simply need to preprocess the data into the JSON lines format.

Download the financial sentiment data file [all-data.csv](https://www.kaggle.com/code/lucamassaron/fine-tune-mistral-v0-2-for-sentiment-analysis/input).

Next, run the [preprocessing script](preprocess/financial_sentiment.py) as follows, updating the global constants INPUT_DATA and OUTPUT to reflect the location of the file downloaded above and the desired output directory, respectively:

```
INPUT_DATA=".../all-data.csv"
OUTPUT=".../sentiment_finance_public"
mkdir -p ${OUTPUT}

python -u financial_sentiment.py \
--input_data_csv ${INPUT_DATA} \
--output_train_jsonl_file ${OUTPUT}/"train.jsonl" \
--output_calibration_jsonl_file ${OUTPUT}/"calibration.jsonl" \
--output_test_jsonl_file ${OUTPUT}/"test.jsonl" \
--output_label_display_names_jsonl_file ${OUTPUT}/"sentiment_3class_labels.jsonl"
```

Next, just upload the document files (and optionally, the label display names file) to Reexpress and train. Feel free to experiment, but a recommended starting point is just to train for 200 epochs, otherwise using the default settings. To compare to the 7 billion parameter generative models, we recommend using the **Fast I** model, which has 3.2 billion parameters. 

And that's all it takes! The overall point accuracy will be similar to fine-tuning the recent 7b generative models mentioned in the Medium post, but with the indispensable additional advantages of uncertainty quantification, interpretability by example/exemplar, and semantic search capabilities. 

Training on a M2 Ultra 76 core Mac Studio with 128gb of RAM took only 21 minutes for training for 200 epochs plus running inference with the **Fast I** model over all 4846 documents. As a point of reference, the point accuracy was 87.11 (784 documents out of 900) on the Test set...but let's not focus on point estimates! In Reexpress, we get reliable calibrated probabilities and can also easily identify the most reliable predictions. Try it out for yourself!
