"""
Download arXiv abstracts via the arXiv search API (based on arXiv id's and/or a simple keyword search)
and format the abstracts for input into Reexpress.

The keyword search is very simple, always using the following convention (here, using the example
'uncertainty quantification,cs.CL'):
"https://export.arxiv.org/api/query?search_query=all:%22uncertainty+quantification%22+AND+cat:cs.CL&start=0&max_results=50"

THIS IS INTENDED TO DOWNLOAD A MODEST NUMBER OF ABSTRACTS to initially cold start the classifier, or to subsequently
add some additional articles that you come across (typically by using the arXiv id's). We use a wait time of 5 seconds
and by default only download a max of 50 articles for each keyword search. 1000-2000 abstracts in the Training
and Calibration sets is typically a reasonable scale with which to start. As noted in the Tutorial, having an initial set of many, many
thousands of abstracts via noisy keywords is typically not what you want, since it'll require post-hoc filtering and
will slow re-training. It's better to start with a modest number of articles and refine over time via the daily RSS
feed, labeling relevant/not relevant articles, and then re-training.

This incorporates example code from the arXiv API manual
(https://info.arxiv.org/help/api/user-manual.html#42-detailed-parsing-examples) by Julius B. Lucks

Most recently tested with:

Python 3.10.9
>>> feedparser.__version__
'6.0.11'
"""

import argparse
import codecs
import json
import uuid
from os import path
import time
import urllib
import random

import feedparser

# Constants
kAPI_wait_time = 5


def save_unprocessed_queries(filename_with_path, lines, is_arxiv_ids_list=False):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for line in lines:
            if len(line) > 0:
                if is_arxiv_ids_list:
                    f.write(line + "\n")
                else:
                    f.write(f"{line[0]},{line[1]}\n")


def save_by_appending_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "a", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def filter_newlines(text_string):
    return " ".join(text_string.split())


def get_query_terms_from_file(filename_with_path, is_arxiv_ids_file=False):
    query_list = []
    with codecs.open(filename_with_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if is_arxiv_ids_file:
                if len(line) > 0 and ("." in line or "/" in line):
                    query_list.append(line)
            else:
                terms = line.split(",")
                if len(terms) == 2 and len(terms[0].strip()) > 0 and len(terms[1].strip()) > 0:
                    query_list.append((terms[0].strip(), terms[1].strip()))
    return query_list


def get_json_list(base_query_url, arxiv_query, label_int, prompt, apply_uuid_marker):

    json_list = []
    print(f"Query: {base_query_url + arxiv_query}")
    response = urllib.request.urlopen(base_query_url + arxiv_query).read()

    # parse the response using feedparser
    feed = feedparser.parse(response)
    for entry in feed.entries:
        title = entry.title
        arxiv_id = entry.id
        arxiv_id = arxiv_id.replace("http", "https", 1)
        original_published_date = entry.published
        most_recent_update = entry.updated
        abstract_text = entry.summary

        if len(entry.authors) == 1:
            authors = f"by {entry.authors[0].name}."
        elif len(entry.authors) <= 3:
            first_authors = entry.authors[0:3]
            last_author = first_authors.pop()
            if len(entry.authors) == 2:  # no comma
                authors = "by " + ", ".join([x['name'] for x in first_authors]) + " and " + f"{last_author['name']}."
            else:
                authors = "by " + ", ".join([x['name'] for x in first_authors]) + ", and " + f"{last_author['name']}."
        else:
            first_authors = entry.authors[0:3]
            authors = "by " + ", ".join([x['name'] for x in first_authors]) + ", et al."

        categories = ", ".join([x["term"] for x in entry.tags[0:2]])
        categories = f"({categories})"

        document = f"{filter_newlines(title)} {filter_newlines(authors)} {categories}\n\nAbstract: {filter_newlines(abstract_text)}"
        if apply_uuid_marker:
            doc_id = f"{arxiv_id} label{label_int} {str(uuid.uuid4())}"
        else:
            doc_id = f"{arxiv_id} label{label_int}"
        json_obj = {"id": doc_id, "label": label_int,
                    "document": document.strip(),
                    "prompt": prompt.strip(),
                    "group": f"Last updated: {most_recent_update}",
                    "info": f"Originally published: {original_published_date}"}
        json_list.append(json_obj)

    if len(json_list) == 0:
        print("ERROR: Search returned no arXiv abstracts. Exiting.")
        print(feed)
        exit()

    return json_list


def main():
    parser = argparse.ArgumentParser(
        description="-----[Download articles from the arXiv search API, constructing the "
                    "JSON lines formatted files for input to Reexpress.]-----")
    parser.add_argument(
        "--arxiv_id_file", default="",
        help="One arXiv id per line, using the arXiv id format since 2007 (e.g., 1503.04069) or the older format from "
             "before March 2007 (e.g., cs/9301115). "
             "--arxiv_id_file and/or --arxiv_topics_file must be provided.")
    parser.add_argument(
        "--arxiv_topics_file", default="",
        help="topic keyword(s) and arXiv category, separated by a comma, one per line. "
             "Avoid non-ascii characters, including "
             "dashes and other punctuation in the keywords. Multi-word keywords should be separated by a single space. "
             "Example: uncertainty quantification,cs.CL. "
             "--arxiv_id_file and/or --arxiv_topics_file must be provided.")
    parser.add_argument(
        "--prompt",
        default="Please classify the relevance of the following article abstract, explaining your reasoning step by step.",
        help="Prompt assigned to each document. In this setting, since you'll be fine-tuning against the supervised "
             "labels with (typically) a non-trivial amount of documents, more important than slight variations in the prompt "
             "is consistently using the same prompt for all documents. Simply using this default prompt across "
             "all arXiv abstracts is the recommended approach.")
    parser.add_argument(
        "--label_int", default=0, type=int,
        help="Label assigned to each document. Use 0 for the 'not relevant' class and 1 for the "
             "'relevant' class, modeling the task as binary classification. For initially cold-starting the "
             "classifier, you will need to run this script twice (at least once for --label_int 0 and at least "
             "once for --label_int 1), with --arxiv_id_file and/or --arxiv_topics_file corresponding to "
             "'not relevant' and 'relevant' articles to download from the arXiv API.")
    parser.add_argument(
        "--output_progress_directory", default="",
        help="The remaining unprocessed id's and/or topics from --arxiv_id_file and/or --arxiv_topics_file will be "
             "saved in files in this directory with the file name formatted as "
             "remaining_arxiv_id_file_label[label_int]_[UUID].txt and/or "
             "remaining_arxiv_topics_file_[label_int]_[UUID].txt. If this script exits with an error, you can "
             "use these files as arguments to --arxiv_id_file and/or --arxiv_topics_file to continue the download.")
    parser.add_argument(
        "--output_training_filename", default="",
        help="JSON lines file (with the required file extension .jsonl) "
             "for use as input to Reexpress. Note that new lines will be *appended* to this file, "
             "rather than recreated.")
    parser.add_argument(
        "--output_calibration_filename", default="",
        help="JSON lines file (with the required file extension .jsonl) for use as input to Reexpress. "
             "This is created as a convenience for cold-starting the classifier. Currently, articles downloaded via "
             "keywords will be placed 50/50 into -output_training_filename and --output_calibration_filename. "
             "Articles downloaded via --arxiv_id_file will all be placed in --output_training_filename. "
             "Note that new lines will be *appended* to this file, rather than recreated.")

    options = parser.parse_args()
    random.seed(0)
    # an initial pause just in case this script is used in a loop in a bash script
    time.sleep(kAPI_wait_time)
    base_query_url = "https://export.arxiv.org/api/query?"
    batch_size = 50
    progress_file_suffix = f"label{options.label_int}_{str(uuid.uuid4())}.txt"

    if options.arxiv_id_file.strip() != "":
        query_arxiv_ids = get_query_terms_from_file(options.arxiv_id_file.strip(), is_arxiv_ids_file=True)
        # de-duplicate
        query_arxiv_ids = list(set(query_arxiv_ids))
        print(f"A total of {len(query_arxiv_ids)} unique arXiv IDs are consider for retrieval.")
        remaining_arxiv_id_file = path.join(options.output_progress_directory.strip(),
                                            f"remaining_arxiv_id_file_{progress_file_suffix}")
        print(f"Remaining arXiv id queries to be processed will be saved to {remaining_arxiv_id_file}")
        save_unprocessed_queries(remaining_arxiv_id_file, query_arxiv_ids, is_arxiv_ids_list=True)
        for i in range(0, len(query_arxiv_ids), batch_size):
            time.sleep(kAPI_wait_time)
            considered_ids_string = ",".join(query_arxiv_ids[i:i + batch_size])
            arxiv_query = f"id_list={considered_ids_string}&start=0&max_results={batch_size}"
            try:
                json_list = get_json_list(base_query_url, arxiv_query, options.label_int, options.prompt, True)
                save_by_appending_json_lines(options.output_training_filename, json_list)
                save_unprocessed_queries(remaining_arxiv_id_file, query_arxiv_ids[i+batch_size:], is_arxiv_ids_list=True)
            except:
                print("An unexpected error was encountered when attempting the following query:")
                print(f"{arxiv_query}")
                print(f"Exiting.")
                exit()

    if options.arxiv_topics_file.strip() != "":
        query_arxiv_topics = get_query_terms_from_file(options.arxiv_topics_file.strip(), is_arxiv_ids_file=False)
        # de-duplicate
        query_arxiv_topics = list(set(query_arxiv_topics))
        print(f"A total of {len(query_arxiv_topics)} unique arXiv topics are consider for retrieval.")
        remaining_arxiv_topics_file = path.join(options.output_progress_directory.strip(),
                                            f"remaining_arxiv_topics_file_{progress_file_suffix}")
        print(f"Remaining arXiv topic queries to be processed will be saved to {remaining_arxiv_topics_file}")
        save_unprocessed_queries(remaining_arxiv_topics_file, query_arxiv_topics, is_arxiv_ids_list=False)
        i = 0
        for arxiv_topic_tuple in query_arxiv_topics:
            time.sleep(kAPI_wait_time)
            escaped_keywords = "+".join([x for x in arxiv_topic_tuple[0].split()])
            escaped_keywords = f"%22{escaped_keywords}%22"  # currently, the keywords are always in quotes
            arxiv_query = f"search_query=all:{escaped_keywords.strip()}+AND+cat:{arxiv_topic_tuple[1].strip()}&start=0&max_results={batch_size}"
            try:
                json_list = get_json_list(base_query_url, arxiv_query, options.label_int, options.prompt, False)
                training_list = []
                calibration_list = []
                if len(json_list) > 2:
                    for one_document_jsonl in json_list:
                        include_in_training = random.randint(0, 1)
                        if include_in_training == 1:
                            training_list.append(one_document_jsonl)
                        else:
                            calibration_list.append(one_document_jsonl)
                else:
                    training_list = json_list

                if len(training_list) > 0:
                    save_by_appending_json_lines(options.output_training_filename, training_list)
                if len(calibration_list) > 0:
                    save_by_appending_json_lines(options.output_calibration_filename, calibration_list)
                i += 1
                save_unprocessed_queries(remaining_arxiv_topics_file, query_arxiv_topics[i:], is_arxiv_ids_list=False)
            except:
                print("An unexpected error was encountered when attempting the following query:")
                print(f"{arxiv_query}")
                print(f"Exiting.")
                exit()


if __name__ == "__main__":
    main()
