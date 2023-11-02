# Example Data
Example data for Reexpress one (macOS application)

The [zip archive](data/sentiment.zip) contains a subset of the commonly used sentiment benchmark dataset. The full dataset is available in typical machine learning benchmark collections, for example [here](https://huggingface.co/datasets/imdb).

Unzip the archive and create a project for binary (2-class) classification. We recommend using the default prompt for sentiment. (There is enough training data that including a prompt is not absolutely required, but it is generally a good practice to include a short prompt to maximize effectiveness, especially if you are bootstrapping from an initially small amount of *labeled* training data.) Add the data to the Training set, Calibration set, Validation set, and Eval set, matching the filenames, via **Data**->**Add**.

To change the label names displayed in the interface from numbers (0,1) to "negative"/"positive", upload the file `label_display_names.jsonl` via **Data**->**Labels**.

Next, go to the **Learn** tab and start training.

That's all that's needed to start the data analysis journey, which includes dense (vector) matching, semantic searches, auto visualizations, uncertainty quantification, and more...all without writing a single line of analysis code nor messing with tricky GPU drivers nor sending your data to an external server!


*Tip: We provide this data only for learning purposes. Since this is a commonly used machine learning benchmark dataset, the eval set is (naturally) not actually a good dataset for held-out evaluation (due to the potential for indirect data contamination). Once you have trained your initial model, try evaluating held-out effectiveness by adding your own reviews. Consider also experimenting with adding reviews with different types of products and different domains (e.g., short form SMS-style social media posts) to examine the capabilities over distribution-shifted documents.*

