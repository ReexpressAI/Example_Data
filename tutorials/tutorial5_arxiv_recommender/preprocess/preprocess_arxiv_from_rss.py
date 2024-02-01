"""
Version 2.0, updated with the new arXiv RSS format as of 2024-02-01.

Download the daily arXiv RSS feed for a given arXiv category (e.g., cs.LG) and format the abstracts for input into
Reexpress.

The id assigned to each resulting 'document' in the JSON lines file is the URL to the current version of the arXiv
article. In Reexpress, you can then use that to pull up the full article by copying and pasting into your browser.

If you want to tweak the document format, or otherwise experiment with variations, we recommend downloading the RSS
file once in your browser, and then supplying the file as an argument via --read_from_file. This will help
ensure you don't hit the arXiv API/servers unnecessarily often. (Remember, if you do change the formatting, you'll
also want to be consistent with the documents you're using for training and calibration, as via
preprocess_arxiv_via_search_api.py.)

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
import feedparser

# Constants
kAPI_wait_time = 5


def save_json_lines(filename_with_path, json_list):
    with codecs.open(filename_with_path, "w", encoding="utf-8") as f:
        for json_dict in json_list:
            f.write(json.dumps(json_dict, ensure_ascii=True) + "\n")


def filter_newlines(text_string):
    return " ".join(text_string.split())


def get_json_list(rss_url, rss_filename, label_int, prompt):
    """
    Download the RSS feed and construct a JSON object for each abstract
    """
    json_list = []
    if rss_url is not None:
        response = urllib.request.urlopen(rss_url).read()
        feed = feedparser.parse(response)
    else:
        feed = feedparser.parse(rss_filename)

    if len(feed.entries) > 0:
        arxiv_category = feed.feed.title.split()[0].strip()
        rss_last_updated = feed.feed.updated
        rss_published = feed.feed.published
    for entry in feed.entries:
        title = entry.title
        # Here we retain the version (i.e., v[Int]) as the id, which means that abstract revisions will not be
        # overwritten. (Note that it is easy to find all revisions for comparison/deletion. Simply run a keyword search using
        # the base arXiv id in Reexpress.) Use arxiv_id = entry.link if you want to use the base arXiv id (and by
        # extension, always overwrite).
        arxiv_id = f"https://arxiv.org/abs/{entry.id[entry.id.rfind(':')+1:]}"

        if entry.arxiv_announce_type in ["new", "cross"]:
            original_published_date = rss_published
            most_recent_update = "n/a"
        else:
            original_published_date = f"unavailable (updated article from RSS feed)"
            most_recent_update = rss_published

        abstract_text = entry.summary

        authors_list = [x.strip() for x in entry.author.split("\n")]
        # As in our script to download from the search API, we only show up to 3 authors by name. Feel free to adjust.
        if len(authors_list) == 1:
            authors = f"by {authors_list[0].strip()}."
        elif len(authors_list) <= 3:
            first_authors = authors_list[0:3]
            last_author = first_authors.pop()
            if len(authors_list) == 2:  # no comma
                authors = "by " + ", ".join(
                    [x.strip() for x in first_authors]) + " and " + f"{last_author.strip()}."
            else:
                authors = "by " + ", ".join(
                    [x.strip() for x in first_authors]) + ", and " + f"{last_author.strip()}."
        else:
            first_authors = authors_list[0:3]
            authors = "by " + ", ".join([x.strip() for x in first_authors]) + ", et al."

        # only the first 2 categories are retained
        categories = ", ".join([x["term"] for x in entry.tags[0:2]])
        categories = f"({categories})"

        document = f"{filter_newlines(title)} {filter_newlines(authors)} {categories}\n\nAbstract: {filter_newlines(abstract_text)}"

        json_obj = {"id": arxiv_id, "label": label_int,
                    "document": document.strip(),
                    "prompt": prompt.strip(),
                    "group": f"Last updated: {most_recent_update}",
                    "info": f"Originally published: {original_published_date}"}
        json_list.append(json_obj)

    if len(json_list) == 0:
        print("ERROR: Search returned no arXiv abstracts. Exiting.")
        print(feed)
        exit()

    return arxiv_category, rss_last_updated, json_list


def main():
    parser = argparse.ArgumentParser(
        description="-----[Download the daily arXiv RSS for a given arXiv category, constructing the "
                    "JSON lines formatted files for input to Reexpress.]-----")
    parser.add_argument(
        "--arxiv_category", default="cs.LG",
        help="A valid arXiv category. Example: cs.LG Another example: cs.CL "
             "You can verify the category by opening the corresponding RSS URL in a browser, or RSS reader. E.g.: "
             "https://export.arxiv.org/rss/cs.LG")
    parser.add_argument('--read_from_file', action='store_true',
                        help="If provided, --input_filename must also be provided. This file is simply the downloaded "
                             "arXiv RSS .xml file. For example, you can go to https://export.arxiv.org/rss/cs.LG "
                             "in the Firefox browser, go to File->Save Page As... and then cs.LG.xml will be the "
                             "file to include. The end result is the same if you supply the file directly or "
                             "download the RSS via this script (i.e., by omitting --input_filename). (We primarily "
                             "include this option in case you want to experiment with modifying the document "
                             "formatting without overburdening the arXiv API/servers.)")
    parser.add_argument(
        "--input_filename", default="",
        help="The downloaded RSS XML file. This is only used if --read_from_file is provided.")
    parser.add_argument(
        "--prompt",
        default="Please classify the relevance of the following article abstract, explaining your reasoning step by step.",
        help="Prompt assigned to each document. In this setting, since you'll be fine-tuning against the supervised "
             "labels with (typically) a non-trivial amount of documents, more important than slight variations in the prompt "
             "is consistently using the same prompt for all documents. Simply using this default prompt across "
             "all arXiv abstracts is the recommended approach.")
    parser.add_argument(
        "--label_int", default=-1, type=int,
        help="Label assigned to each document. Typically for the download from the RSS feed, we will use the "
             "unlabeled (-1) value.")
    parser.add_argument(
        "--output_directory", default="",
        help="The JSON lines files (with the required file extension .jsonl) will be placed in this directory with "
             "the following filename convention: [arXiv Category]_[date in the RSS file]_[UUID].jsonl. This "
             "ensures unique filenames if you want to run a script/application (such as Apple's Automator app) to "
             "auto run this script every day.")

    options = parser.parse_args()

    if options.read_from_file:
        if options.input_filename.strip() == "":
            print("You selected the option to read the RSS xml from file, but the filename is blank. Exiting.")
            exit()
        rss_url = None
        rss_filename = options.input_filename.strip()
    else:
        if options.arxiv_category.strip() == "":
            print("The arXiv category is blank. Exiting.")
            exit()
        # an initial pause just in case this script is used in a loop in a bash script
        time.sleep(kAPI_wait_time)
        rss_url = f"https://rss.arxiv.org/rss/{options.arxiv_category.strip()}"
        rss_filename = None
    try:
        parsed_arxiv_category, rss_date, json_list = get_json_list(rss_url, rss_filename, options.label_int,
                                                                   options.prompt)
        save_json_lines(path.join(options.output_directory,
                                  f"{parsed_arxiv_category}_{rss_date}_{str(uuid.uuid4())}.jsonl"),
                        json_list)
    except:
        print(f"An unexpected error was encountered when attempting to download the arXiv RSS feed for the "
              f"following arXiv category: {options.arxiv_category.strip()}")
        print(f"Exiting.")
        exit()


if __name__ == "__main__":
    main()
