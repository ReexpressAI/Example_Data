# Tutorial 7: Reexpress is your reliable co-pilot for enterprise and professional AI-augmented data analysis

### Video overview: [Here](https://youtu.be/ipSZf3h8vLY)

[![Watch the YouTube video](assets/Reexpress_overview.png)](https://youtu.be/ipSZf3h8vLY)


As we saw in Tutorial 6, [Reexpress](https://re.express/) can be used to add uncertainty quantification, search, and auto-visualization capabilities to essentially any generative AI language model. This newfound ability to derive reliable and robust uncertainty estimates from neural networks, over high-dimensional inputs, is very powerful and general. Progress can be made on challenging tasks such as fact verification and hallucination detection by casting them as classification for input into Reexpress. Importantly, in addition to being an end-to-end, no-code solution that you can download and use today on your Mac, the underlying Reexpress method is more advanced and reliable than more recently proposed alternatives in the academic literature, as we illustrate in this tutorial.

Recent academic work ([[1], [2], inter alia.](#references)) has sought to extract latent signals within parametric neural networks that are correlated with truthfulness. The goal is to construct a semi- or distantly-supervised classifier over these signals for use in reducing hallucinations and other errors from large language models. The methods are premised on the idea that large networks encode signals for binary truthfulness in their hidden states, and these latent signals can be extracted via a transform of the hidden states.

**In fact, the older idea that is Reexpress is more powerful and general than this.** Reexpress is premised on our older and more general result that parametric neural networks can be closely approximated as a transformation of their hidden states over the observed training data. Reexpress takes this line of work to its logical conclusion: By reexpressing a model (or a composition of models, as we'll do in this tutorial) as a direct connection between the observed training and calibration sets, we can add the missing properties needed to use neural networks in practice:

- Introspection
  - Interpretability by example/exemplar: The parameters of the network(s) are non-identifiable, but we can relate predictions to instances with known labels in feature space. 
- Updatability
  - Localized modifications without a full re-training via direct label modification in Training and Calibration. (This is orthogonal to, and can be used in conjunction with, retrieval and localized representation fine-tuning.)
- Uncertainty
  - Reliable and robust uncertainty estimates by controlling for Similarity ⧉ Distance ⧈ Magnitude ⧇ data partitions.

In this tutorial, we focus in particular on the Uncertainty property of Reexpress, with an emphasis on its robustness to distribution shifts across the data partitions. A small example is sufficient to demonstrate that Reexpress is not only convenient given its no-code, visual interface, but also that it is essential in enterprise and professional settings.

## Fact verification

Task:

> Determine whether a statement is true or false.

We will use the SAPLMA data from [1], which consists of single sentence statements that the model is then tasked with classifying as true or false. A final held-out test set is then constructed by having an LLM generate a statement continued from a true statement not otherwise in the dataset. These test statements are checked manually and assigned labels by human annotators. An important observation in their paper is that accuracy is dramatically lower on this generated, held-out test set. That is, the held-out test set is distribution-shifted relative to the training data. Distribution-shifts are common in real-world applications, and as with this data, it is not always immediately obvious to the naked eye that the data is substantively different than the training data. In this case, the test sentences would seem to also be simple true/false statements, but instead the model goes off the rails on this distribution.

If we fine-tune (or re-calibrate) based on a new distribution, we can typically improve accuracy over new samples from the new distribution. However, that begs the obvious question: How would we even know the new unseen data is distribution-shifted in the first place since we wouldn't have ground-truth labels? (After all, we're using the model to predict over this new data!) *The answer is you wouldn't know unless you had Reexpress.* This is why Reexpress is a dramatically more powerful (and fundamentally different) idea than alternative approaches in the literature.

Let's look at this in practice.

## Training Reexpress for fact verification

As in [Tutorial 6](/tutorials/tutorial6_mlx/README.md), we compose the on-device model (in this case, **Fast I**, a multi-lingual model with 3.2 billion parameters) with attributes derived from a simple re-ask verification over `Mixtral-8x7B-Instruct-v0.1`. See [preprocess/README.md](preprocess/README.md) for how to run inference with the Mixtral model and format the attributes in the JSON lines files. That basic setup can be used for other classification tasks, so you can use that as a starting point for your own data. However, for the purposes of this tutorial, you can skip the preprocessing step, as the preprocessed data with attributes is available for download in the archive [data/saplma_true_false_data_with_mixtral_instruct_v0.1.zip](data/saplma_true_false_data_with_mixtral_instruct_v0.1.zip).

Create a new project in Reexpress. Choose the **Fast I** (3.2 billion parameter) model and use the following prompt: 

> Please classify the correctness of the information in the following document.

The basic pattern here holds for any random split of the data. For reference, the split used here is the following:

| Filename | Datasplit |
| --- | --- |
| `cities_true_false.csv.attributes.jsonl` | Training |
| `elements_true_false.csv.attributes.jsonl` | Training |
| `facts_true_false.csv.attributes.jsonl` | Training |
| `inventions_true_false.csv.attributes.jsonl` | Calibration |
| `companies_true_false.csv.attributes.jsonl` | Calibration |
| `animals_true_false.csv.attributes.jsonl` | Calibration |

The held-out eval set is `generated_true_false.csv.attributes.jsonl`. (In the video, and below, we've renamed this Eval set to "Distribution-shifted (unseen) test". To rename a datasplit in Reexpress, go to **Data** and click *Rename* on the applicable datasplit.)

After importing the above files, optionally upload the label display names file, `label_dislay_names.jsonl`.

Click **Learn**->**Train**->**Batch** and choose 200 epochs for `Model training` and `Model compression`. Scroll down and select the Eval set to automatically run post-training inference. And then start! This will take around 30 minutes on an M2 Ultra 76-core GPU with 128 GB of memory. (Expect around 2x longer on an M1 Max with 64 GB of memory.)


## Analyzing the output

Go to **Compare** and **Select** the Calibration set. On our run, the Balanced Accuracy is 86, which seems promising given the non-triviality of the task. Great. Let's now evaluate our held-out test. Oh, no! As with the paper [1], the test accuracy has dropped dramatically to the 60s, as displayed in the screenshot below (Calibration set, *left*, Eval set, *right*).

![Unconstrained eval on the test set](assets/unconstrained.png)

As in [1], one approach would be to fine-tune/re-calibrate over this new distribution. However, that wouldn't in itself be sufficient in real settings, since we would then be caught off guard on the next distribution shift.

Luckily, with Reexpress we can altogether avoid being caught off guard in the first place. And we don't even have to write any additional code to enable this critical guardrail! The calibration method in Reexpress is robust to distribution shifts across the data partitions, and we can utilize the second-order estimate of calibration reliability to further restrict to the data over which the calibration process itself is most reliable. For example, we can restrict to the documents with a calibrated probability in [0.95, 0.99] among those in the Highest calibration reliability partition:

![Select the documents with probability >= 95% within the highest calibration reliability partition](assets/partition_selection.png)

Unlike the massive discrepancy in accuracy between the Calibration and the Test set that we saw earlier (upper 80s to low 60s), we now see that the selected subsets have a similarly high accuracy. This reflects that this subset of the data is well calibrated: 

![Constrained eval on the test set](assets/constrained_selection.png)

What about the remaining documents? The remaining documents are those for which we need to more carefully examine, and if applicable, send to a human expert for adjudication. An actionable next step would be to label some of those documents, and then use those newly labeled documents to update the model by adding them to the Training and Calibration sets.


## Additional data analysis

With Reexpress, analysis tasks over natural language text that were previously viewed as difficult and programming intensive, if not intractable, become trivially easy via the no-code environment. Above we saw that it was easy to detect and control for distribution shifts and to obtain well-calibrated probabilities. We can additionally run semantic searches and keyword searches, with or without constraints based on the data partitions, to further examine the data.

A quick examination of the data reveals that not all of the test set is on a disjoint topic, despite care taken to achieve that structure when creating the dataset. While there are no exact matches, there are some sentences with topic overlap in `generated_true_false.csv` vs. the remaining documents. For example (true label in brackets):

Test:

>Canberra is located in Australia. {True}

Training:

>Canberra is a name of a city. {True}

>Canberra is a name of a country. {False}

>Canberra is a city in Australia {True}

There also some sentences in the dataset with unusual phrases, such as the following in Calibration, which has a label of `True`.

>The eagle has a habitat of various.

Datasets are difficult to create and curate, especially at scale. Reexpress greatly simplifies the process of uncovering data quality errors and improving the quality of your datasets. This in turn improves the quality of your downstream decision-making based on your model's outputs.


## Concluding remarks

In this tutorial, we've demonstrated how Reexpress is a more powerful and practical approach than more recent academic proposals for uncovering latent signals from neural networks. Reexpress provides critical properties not available anywhere else: Robust calibrated probabilities; out-of-distribution detection; interpretability via a direct connection between new unlabeled test documents and the data with known labels; and advanced search capabilities. Amazingly, unlocking these capabilities for *your data* doesn't require writing a single additional line of analysis code.

In this tutorial, we've used a simplified version of the more general task of detecting hallucinations/errors in generative AI output. Here, the emphasis is on deriving uncertainty estimates over latent knowledge within the parameters of the networks themselves. In some use-cases, we have access to additional external signals, such as via retrieval from a document database or a web query. Reexpress can readily be applied to these cases, as well. Simply incorporate the output from retrieval in the document text and/or as attribute vectors. The overall setup is otherwise the same as we've seen here. In this way, Reexpress can be used as the final verification/classification layer for essentially any natural language task.

## References

[1] Azaria and Mitchell. 2023. "The Internal State of an LLM Knows When It’s Lying". [https://arxiv.org/abs/2304.13734](https://arxiv.org/abs/2304.13734)

[2] Burns et al. 2022. "Discovering Latent Knowledge in Language Models Without Supervision". [https://arxiv.org/abs/2212.03827](https://arxiv.org/abs/2212.03827)
