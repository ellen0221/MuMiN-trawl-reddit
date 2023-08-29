'''Fetch Twitter data related to the keywords of interest'''
import json

import pandas

from .translator import GoogleTranslator
# from .twitter import Twitter
from .utils import root_dir
from tqdm.auto import tqdm
from typing import List
import logging
import yaml
from keybert import KeyBERT
import pandas as pd
import datetime as dt
from pathlib import Path
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv;

load_dotenv()
import os
import praw

logger = logging.getLogger(__name__)


def fetch_reddits(keywords: List[str] = ['coronavirus', 'covid'],
                  translate_keywords: bool = False,
                  translate_to_en: bool = False,
                  max_results_per_query: int = 100_000,
                  reddit_data_dir: str = 'data/reddit',
                  progress_bar: bool = True,
                  output_dir_prefix: str = '',
                  claim_idx: int = 0,
                  reddit: praw.Reddit = None,
                  **kwargs):
    '''Fetch all the data from Reddit related to the keywords of interest.

    The results will be stored in the 'data/reddit' folder.

    Args:
        keywords (list of str, optional):
            List of keywords to search for. These keywords will be translated
            into all the BCP47 languages and query'd separately. Defaults to
            ['coronavirus', 'covid'].
        translate_keywords (bool, optional):
            Whether to translate the `keywords` into all BCP47 languages.
            Defaults to False.
        max_results_per_query (int, optional):
            The maximal number of results to get per query. Must be at least
            10. Defaults to 100_000.
        reddit_data_dir (str, optional):
            The (relative or absolute) path to the directory containing all the
            reddits. Defaults to 'data/reddit'.
        progress_bar (bool, optional):
            Whether a progress bar should be shown or not. Defaults to
            True.
        output_dir_prefix (str, optional):
            A prefix to be prepended to the output directory name for each
            keyword, meaning the folder names will be of the form
            "<output_dir_prefix><keyword>". Defaults to an empty string.
        claim_date (str, optional):
            The datetime of the claim. This is used for filtering research result.
            If this field is not specified, the research result will not be filtered.
            Defaults to ''.
        **kwargs:
            Keyword arguments to use in Reddit API.
    '''
    # Search from all subreddit
    all_subreddit = reddit.subreddit("all")

    # Initialise the list of queries
    query_list = list()

    # Include the keywords and all their translations
    if len(keywords) > 0:

        if translate_to_en:
            # Translate keywords to English
            translator = GoogleTranslator()
            translated_keywords = translator(keywords, target_lang="en")
            if isinstance(translated_keywords, str):
                translated_keywords = [translated_keywords]

            keywords = translated_keywords

        # Add keywords to the list of queries
        query_list.extend(keywords)

        if translate_keywords:
            # Initialise translation wrapper
            translator = GoogleTranslator()

            # Load list of BCP47 languages
            bcp47_path = root_dir / 'data' / 'bcp47.yaml'
            bcp47 = [lang for lang in yaml.safe_load(bcp47_path.read_text())
                     if isinstance(lang, str)]

            # Make list of keywords
            query_list_en = query_list.copy()
            for language_code in tqdm(bcp47, desc='Translating keywords'):
                translated_queries = translator(query_list_en,
                                                target_lang=language_code)
                if isinstance(translated_queries, str):
                    translated_queries = [translated_queries]
                query_list.extend(translated_queries)

    # Remove duplicates
    # query_list = list({q.lower() for q in query_list})

    # Ensure that only strings are in the query list
    query_list = [q for q in query_list if isinstance(q, str)]

    # Define directory with all Reddit data
    if reddit_data_dir.startswith('/'):
        reddit_data_dir = Path(reddit_data_dir)
    else:
        reddit_data_dir = root_dir / reddit_data_dir

    # Create Reddit directory if it does not already exist
    if not reddit_data_dir.exists():
        reddit_data_dir.mkdir()

    # Initialise progress bar
    if progress_bar:
        pbar = tqdm(total=len(query_list))

    # Fetch loop
    for query in query_list:
        print(f'Processing claim{claim_idx}: {query}')

        # Update progress bar description
        if progress_bar:
            pbar.set_description(f'Fetching reddits related to "{query}"')

        # Set up query subdirectory
        # query_folder_name = query.replace('#', 'hashtag_').replace(' ', '_')
        query_folder_name = "claim" + str(claim_idx)
        # query_folder_name = output_dir_prefix + query_folder_name
        query_dir = reddit_data_dir / query_folder_name
        new_query_folder_name = output_dir_prefix + '-' + query_folder_name
        new_query_dir = reddit_data_dir / new_query_folder_name

        # Skip to next query if its directory already exists
        if query_dir.exists():
            query_dir.rename(new_query_dir)
            # Update the progress bar
            if progress_bar:
                pbar.update(1)
            continue
        elif new_query_dir.exists():
            # Update the progress bar
            if progress_bar:
                pbar.update(1)
            continue

        query_folder_name = output_dir_prefix + '-' + query_folder_name
        query_dir = reddit_data_dir / query_folder_name

        # Fetch the Reddit data
        try:
            results = all_subreddit.search(query, limit=50)
        except Exception as e:
            print(f'Error when processing claim{claim_idx}: "{e}"')
            continue

        # Save the Reddit data
        data_list = []
        for submission in results:
            submission_dict = submission.__dict__
            # Handle posts with plain text or link
            if hasattr(submission, "selftext") or (
                    hasattr(submission, "post_hint") and 'video' not in submission.post_hint):
                # Select specific fields
                limited_field_dict = {key: submission_dict[key] for key in submission_dict.keys() &
                                      {'id', 'name', 'title', 'post_hint', 'selftext', 'url', 'num_comments',
                                       'ups', 'created_utc', 'author_fullname', 'subreddit_name', 'subreddit_id',
                                       'num_crossposts', 'subreddit_name_prefixed'}}
                # Save the top 15 relevant posts into data_list
                if len(data_list) < 15:
                    data_list.append(limited_field_dict)
                else:
                    break

        if len(data_list) > 0:
            # Create the query directory
            query_dir.mkdir()
            path = query_dir / f'claim{claim_idx}.csv'
            df = pandas.DataFrame.from_records(data_list)
            df.to_csv(path)
        else:
            print('No post after filtering!')

        # Update the progress bar
        if progress_bar:
            pbar.update(1)


def fetch_reddits_from_claims(
        max_results_per_query: int = 1_000,
        gpu: bool = True,
        reddit_data_dir: str = 'data/reddit'):
    '''Extract keywords from all claims and fetch reddits for each of them.

    Args:
        max_results_per_query (int, optional):
            The maximal number of results to get per query. Must be at least
            10. Defaults to 1_000.
        gpu (bool, optional):
            Whether to use the GPU for keywords extraction. Defaults to True.
        reddit_data_dir (str, optional):
            The (relative or absolute) path to the directory containing all the
            tweets. Defaults to 'data/reddit'.
    '''

    # Initialise the keyword model
    # transformer = 'paraphrase-multilingual-MiniLM-L12-v2'
    # sbert = SentenceTransformer(transformer, device='cuda' if gpu else 'cpu')
    # kw_model = KeyBERT(sbert)
    # kw_config = dict(top_n=3, keyphrase_ngram_range=(1, 2))

    # Initialise a praw.Reddit instance
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        refresh_token=os.getenv('REDDIT_REFRESH_TOKEN'),
        user_agent=os.getenv('REDDIT_USER_AGENT'),
    )

    # Iterate over all the fact-checking websites
    reviewers = root_dir / 'data' / 'reviewers'
    desc = 'Fetching reddits'
    for reviewer_csv in tqdm(list(reviewers.iterdir()), desc=desc):

        # Load the CSV file with all the claims for the given reviewer
        reviewer_df = pd.read_csv(reviewer_csv)

        # Filter out claims whose predicted verdict is other
        reviewer_df = reviewer_df[['other' not in r['predicted_verdict']
                                   for _, r in reviewer_df.iterrows()]]

        # Remove punctuation and numbers from the claims in the dataframe
        regex = r'[.“\"]'
        reviewer_df['claim'] = (reviewer_df.claim
                                           .str
                                           .replace(regex, '', regex=True))

        # Convert date column to datetime types
        # reviewer_df['date'] = pd.to_datetime(reviewer_df.date, errors='coerce')
        # reviewer_df.dropna(subset=['date'], inplace=True)

        # Extract the keywords from the claims
        # keywords = [{kw[0]
        #              for kw in kw_model.extract_keywords(claim, **kw_config)}
        #             for claim in reviewer_df.claim]

        claims = reviewer_df['claim'].values.tolist()

        desc = f'Fetching reddits for {str(reviewer_csv.stem)}'
        pbar = tqdm(claims, desc=desc)
        for (index, row), kw_set in zip(reviewer_df.iterrows(), pbar):
            translate_to_en = True if row.language != "en" else False
            fetch_reddits(keywords=[kw_set],
                          max_results_per_query=max_results_per_query,
                          translate_to_en=translate_to_en,
                          reddit_data_dir=reddit_data_dir,
                          progress_bar=False,
                          output_dir_prefix=str(reviewer_csv.stem),
                          claim_idx=int(index),
                          reddit=reddit)
