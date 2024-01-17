# Tutorial 5: Taming the arXiv Deluge with a Personalized Article Recommender

Video overview: [Here](https://youtu.be/k1H3GcDdAfs)

[Reexpress](https://re.express/) works remarkably well as a *personalized* arXiv article recommender/filter. This can save scholars a lot of time in practice, since the number of articles in the daily RSS feeds in AI related arXiv categories (e.g., cs.CL, which contains Natural Language Processing papers, and cs.LG for Machine Learning) has exploded in recent years. (In this tutorial, we use the cs.CL and cs.LG categories, but this works for any of the arXiv categories.)

To create a personalized recommendation classifier, we model the task as binary classification, with the following semantics for the two labels: Not relevant (label 0) and relevant (label 1). Once we have trained the model, we can then focus our attention on the papers with the highest calibration probability/reliability. And over just a few days, you can rapidly improve your recommendations by marking any new papers you read as relevant/not relevant and then clicking **retrain** without writing any code. Want to find that paper you vaguely remember seeing and marking as relevant in the RSS a few weeks earlier? Just use the no-code doc-level and feature-level dense search, and/or the no-code semantic search. 

*NOTE: It's important to remember that the purpose of the classifier is to winnow the set of daily arXiv abstracts to those on topics of interest to you. This relies on the surface-level characterstics of the text, which is exactly what's needed for this purpose. The classifier does not seek to determine good/bad research, nor reliable/unreliable research. That of course is not possible, in general, only via the abstract. Further, we have, by design, in the current version of the preprocessing scripts, not processed the Comments attributes of the article metadata, which sometimes contain indicators of conference or journal acceptance. The goal here is to surface topically relevant articles as they are released in order to inform one's own research and to engage authors in timely constructive feedback and discussions.*

It's quite easy to build your own classifier. We just need to download an initial modest group of arXiv abstracts to cold start the classifier, and then each day we can download and process the RSS feed, importing the result into the no-code Reexpress app. We provide simple scripts here to handle the arXiv downloads and formatting.

**You won't need to write any new code yourself.**

## Step 1: Cold-start the classifier

We'll need a modest number of articles to initially train the model. We include a simple script here that will download and format arXiv abstracts based on a set of arXiv identifiers (e.g., `cs/9301115` or `1503.04069`) and/or keyword topics + arXiv category (e.g., `uncertainty quantification,cs.CL`) that you provide in input files to the script. The articles retrieved from arXiv identifiers will be saved directly to the Training set (under the assumption that these are articles you have more carefully curated), and those from keywords will be split equally across the Training and Calibration sets. (If you only want to use the articles downloaded via arXiv identifiers, you'll need to manually split the generated Training file into a Training and Calibration set.) The articles retrieved from keywords will be noisy, but that's fine; you can easily refine the labels over time within Reexpress based on searching; the predictions; similarity and distance to other articles; etc. *That said, your initial inclination may be to download many, many thousands of articles via keywords. Resist this temptation, as you will find yourself doing a lot of post-hoc filtering; it will make retraining slower (which you'll want to do regularly, even daily, to update the model based on new articles from the RSS feed); and it's unnecessarily heavy on the arXiv API. Aim for, say, a max of 1,000-2,000 articles in each of Training and Calibration to start.*

### Setup and Python script requirements

The Python scripts for formatting are very simple. (All of the serious heavy-lifting is done by the no-code Reexpress app.) Here we'll walk through exactly how to set this up and run in case you're more familiar with Swift or other languages, or haven't used the Mac Terminal application. (We're being a bit vebose for those not familiar with the command line and/or Python, but don't let the length of this Readme dissuade you. --- It's literally just running one Python script to download an initial set of articles, and another script to download the daily RSS feed. **You won't need to write any new code yourself.**)

First, setup a Conda environment. If you do not have Conda installed, see https://conda.io/projects/conda/en/latest/user-guide/getting-started.html (This is not strictly required, but just simplifies managing your Python packages. Once installed, you can just run the following code as-is, without affecting your other Python environments.)

In the Mac Terminal application, create a Conda environment with Python 3.10.9, and install the dependencies [feedparser](https://feedparser.readthedocs.io/en/latest/) and [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/):

```
conda create -n "arxiv" python=3.10.9
conda activate arxiv
pip install feedparser
pip install beautifulsoup4
```

(For reference, we have most recently tested these scripts with feedparser version 6.0.11 and beautifulsoup4 version 4.12.2.)

### Run the script

First, `cd` into the directory containing the scripts preprocess_arxiv_via_search_api.py and preprocess_arxiv_via_search_api.py. You can find this in the Mac Finder and then drag and drop the directory into the Terminal, adding the command `cd` before the directory. The result will look something like the following:

```
cd /Users/your_user_name/Documents/ ... wherever you've saved the repo .../Example_Data/tutorials/tutorial5_arxiv_recommender/preprocess
```

Next, run the script. Here, we are using the default arXiv id's and topics files in the [preprocess/example_input](preprocess/example_input) directory, but you should update those files with whatever articles and keywords are of interest to you. Don't forget that to train the classifier, you'll need to supply both RELEVANT and NOT RELEVANT examples. For the keywords file, avoid non-ascii characters, including dashes and other punctuation in the keywords. Multi-word keywords should be separated by a single space, and an arXiv category needs to be provided. Example: `uncertainty quantification,cs.CL`. Feel free to change the `OUTPUT` directory to a preferred location.

```
conda activate arxiv

OUTPUT="arxiv_tutorial_output"
mkdir -p "${OUTPUT}/running_progress"

python -u preprocess_arxiv_via_search_api.py \
--arxiv_id_file="example_input/relevant_arxiv_ids.txt" \
--arxiv_topics_file="example_input/relevant_arxiv_topics.txt" \
--label_int 1 \
--output_progress_directory "${OUTPUT}/running_progress" \
--output_training_filename "${OUTPUT}/train_relevant.jsonl" \
--output_calibration_filename "${OUTPUT}/calibration_relevant.jsonl"

# Similarly, process the 'not relevant' abstracts. Here, we'll only use a keywords file:

python -u preprocess_arxiv_via_search_api.py \
--arxiv_id_file="" \
--arxiv_topics_file="example_input/not_relevant_arxiv_topics.txt" \
--label_int 0 \
--output_progress_directory "${OUTPUT}/running_progress" \
--output_training_filename "${OUTPUT}/train_not_relevant.jsonl" \
--output_calibration_filename "${OUTPUT}/calibration_not_relevant.jsonl"
```


### A note on id's in the JSON lines file

In Reexpress, the document id defines the uniqueness of a document. Re-uploading a document with the same id will delete any data previously associated with that document and will automatically transfer it to the datasplit chosen when re-uploading. 

When downloading articles from the RSS feed (using `preprocess_arxiv_from_rss.py`), the JSON id is set as the URL to the abstract, e.g., `https://arxiv.org/abs/1503.04069`.

When using `preprocess_arxiv_via_search_api.py`, articles downloaded via --arxiv_id_file are assigned an id with the following format: `[URL] label[LABEL_INT] [UUID]`. Those downloaded via --arxiv_topics_file are assigned an id with the following format: `[URL] label[LABEL_INT]`. We add the `[UUID]` to the former as we assume these are articles you have more carefully curated and don't want them to be auto reassigned to another datasplit if a duplicate is downloaded via a topic keyword. We take this more conservative approach (which could create duplicates) because it's trivially easy to remove duplicates in training (including among those documents already in training) by going to **Explore**->**Select**->**Constraints** and setting Lowest and Highest allowed Distance to 0.0. (This distance is the L2 distance to the Training set.) If any duplicates show up, you can delete them individually (see Delete at the bottom of the document view in Explore) or click **Details**->**Batch** to run a batch delete.


## Step 2: Import to Reexpress and train

Next, create a new project in Reexpress, choosing binary classification. We ourselves use the Faster I (1.2 billion parameter) model for our arXiv recommender, but if you have an older Mac, the FastestDraft I (640 million parameter) model can also be used. The Fast I (3.2 billion parameter) model is mostly overkill for this task, so you might as well choose a smaller model to save time since you'll want to reguarly retrain.

For the `Default prompt` choose custom and copy and paste the following into the text box: 

```
Please classify the relevance of the following article abstract, explaining your reasoning step by step.
```

The Python scripts automatically add this prompt to the JSON for each document, but having a default prompt can save you time when running semantic searches, since you can then just click **Default** to auto-fill the prompt.

Once created, go to **Data**->**Add** and import the train_relevant.jsonl and train_not_relevant.jsonl files to the Training set, and the calibration_relevant.jsonl and calibration_not_relevant.jsonl files to the Calibration set.

Go to **Learn**->**Train**->**Batch**. Training for 200 epochs, otherwise using the default settings, is a reasonable start if (as in the example above) you have rougly 2,000 articles in each of Training and Calibration.

You can change the label diplay names by going to **Data**->**Labels** and importing the file in this repo [resources/label_display_names.jsonl](resources/label_display_names.jsonl).

And that's it. You now have your initial article recommender!

## Step 3: Download and process the daily RSS feed

Now, every day you can just preprocess the RSS feed with the following script. (We can further automate this, as noted below.) Replace "cs.LG" with whatever arXiv category you are interested in, and simply run the following, choosing your desired output directory. The resulting file (if there is indeed an RSS announcement that day) can then simply be imported to Reexpress.

```
conda activate arxiv

OUTPUT="arxiv_rss_output"
mkdir -p "${OUTPUT}"

python -u preprocess_arxiv_from_rss.py \
--arxiv_category="cs.LG" \
--output_directory "${OUTPUT}"
```

### Daily auto download

We can use the Mac Automator application to set up a recurring process to run the `preprocess_arxiv_from_rss.py` script every day there is a new RSS feed. (Check out the Tutorial video. It's easier to see this visually: It's pretty quick to setup.)

Open Automator. New document, choose Application.

Search for Run Shell Script and drag that onto the main canvas.

For Shell, choose /bin/zsh, and then add the following, updating the paths (in square brackets) to wherever you saved the repo script and where you would like to save the output files. Note that if you subsequently change the file locations, this will need to be updated:

```
source $HOME/.zshrc
conda activate arxiv

cd [path_to_repo]/Example_Data/tutorials/tutorial5_arxiv_recommender/preprocess

OUTPUT="arxiv_rss_output"

python -u preprocess_arxiv_from_rss.py \
--arxiv_category="cs.LG" \
--output_directory="${OUTPUT}" > "${OUTPUT}"/csLG_rss_download_log_$(date +%F).txt

python -u preprocess_arxiv_from_rss.py \
--arxiv_category="cs.CL" \
--output_directory="${OUTPUT}" > "${OUTPUT}"/csCL_rss_download_log_$(date +%F).txt

```

(Here we are also saving the standard output to a file with the date to serve as a log. This can be useful to have as a record that a download was attempted but failed. Not that on days when there are no paper releases, the script will exit without saving a .jsonl file. If this is unexpected, you can double check if there is an RSS feed for the day by opening the RSS URL in the Firefox browser. On rare occasions, there are delays in the release. Also, see the note below on the arXiv release schedule.)

Save this application to your hard drive. Run it at least once (by double clicking on the file) to check that it works as expected. Next, we can schedule this to run daily via Apple's Calendar app. That way you won't have to manually run it every day.

Open Calendar and create a New Event. Click Alert->Custom and then choose 'Open file', select the automator app file (in the second drop-down where it will say Calendar by default), and choose 'At time of event'. You can then choose a time for it to repeat every day when there is a new RSS feed. Note that when it first runs, you may get a popup to confirm access to the output directory you specified above.

When does the RSS become available? See [Availability of submissions](https://info.arxiv.org/help/availability.html) and adjust accordingly for your time zone. On days without announcements, the `preprocess_arxiv_from_rss.py` script will exit without saving a .jsonl file.

## Tips

After initially cold-starting the classifier via keywords, the training data may be somewhat noisy. For reference, using the example ids and keywords above, we ended up with around 1500 documents in each of the Training set and Calibration sets, and training resulted in a Calibration set balanced accuracy of around 85. In the Calibration set, there were *no* documents in the Highest or High calibration reliability partitions.

We recommend the following filtering until you've built up a strong classifier: For the day's new RSS feed, focus on the following subset: Go to **Explore**->**Select** and make the following selections:

- Calibrated Probability of Prediction in [0.90, 0.99]
- f(x) Magnitude: High
- Distance to Training: Near AND Far
- Similarity to Training: High AND Medium
- Predicted class: Relevant

Then go to the **Sorting** tab and sort by 'similarity'.

If that results in no returned documents in the day's RSS feed, there may not be any directly relevant articles. But if you've just created the classifier, you may want to look closer. Consider resetting the selection, and then go back to **Explore**->**Select**, and only look at the class 1 predictions:

- Predicted class: Relevant

Again, sort by 'similarity'.

Remember, as you flip through the abstracts, you can click "Label" in the document details and choose "Relevant/Not Relevant" and transfer to the Training and/or Calibration sets. It doesn't take many days to accumulate a fairly large amount of data -- and by extension, a better classifier.

One more tip:

- Recall that in Reexpress the document 'id' defines the uniqueness of the document. If you're uploading multiple RSS feeds that often have overlapping articles (such as cs.LG and cs.CL), duplicate abstracts will be overwritten if an abstract with the same arXiv id is re-imported. In practice, if you plan to import multiple arXiv feeds from the same day, it's a good idea to import all of the files for the day before you start reading, just so that any new labels (or datasplit moves) aren't inadvertently overwritten.

## License

Finally, as a reminder, arXiv abstracts, as with other arXiv metadata, have a Creative Commons CC0 1.0 Universal Public Domain Dedication. As such, if you've curated a particularly informative set of abstracts that you think others in your subfield would find useful to bootstrap their own classifiers, feel free to share them. However, importantly, the articles themselves (including the PDFs) have their own licenses and you cannot necessarily re-host those files directly. Takeaway: Typically you should only distribute the metadata (id, title, abstract, URL, etc.) and not the PDFs or the LaTeX. See [arXiv License Information](https://info.arxiv.org/help/license/index.html#metadata-license) for additional details.

*We thank arXiv for use of its open access interoperability.*
