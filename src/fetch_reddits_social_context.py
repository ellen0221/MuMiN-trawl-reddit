'''Fetch Twitter data related to the keywords of interest'''

import pandas

from .utils import root_dir
from tqdm.auto import tqdm
import logging
import pandas as pd
from pathlib import Path
import praw
import glob
from dotenv import load_dotenv;

load_dotenv()
import os

logger = logging.getLogger(__name__)


def fetch_social_context(reddit: praw.Reddit = None,
                         reddit_df: pandas.DataFrame = None,
                         reddit_data_dir: Path = '',
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
        **kwargs:
            Keyword arguments to use in Reddit API.
    '''

    comment_list = []
    reddit_df = reddit_df.dropna(subset=['num_comments'])

    for _, row in tqdm(reddit_df.iterrows(), desc='Processing each reddit post\'s social context data'):

        # Fetch subreddit data
        # if row['subreddit_name_prefixed'][:2] == 'r/':
        #     # Check if subreddit data exist
        #     subreddit_dir = root_dir / 'data' / 'subreddits'
        #     if not subreddit_dir.exists():
        #         subreddit_dir.mkdir()
        #     subreddit_file = subreddit_dir / f'{row["subreddit_name_prefixed"][2:]}.csv'
        #     if not subreddit_file.exists():
        #         subreddit = reddit.subreddit(row['subreddit_name_prefixed'][2:])
        #         # print(row['subreddit_name_prefixed'][2:])
        #         subredditId = subreddit.id  # Fetch subreddit data
        #         subreddit_dict = subreddit.__dict__
        #         subreddit_field = {key: subreddit_dict[key] for key in subreddit_dict.keys() &
        #                            {'id', 'name', 'title', 'public_description', 'subscribers',
        #                             'display_name_prefixed', 'created_utc'}}
        #         path = subreddit_dir / f'{row["subreddit_name_prefixed"][2:]}.csv'
        #         subreddit_df = pandas.DataFrame.from_records([subreddit_field])
        #         subreddit_df.to_csv(path, index=False)

        # Only process populate posts
        if row['num_comments'].isdigit() and row['num_comments'] >= 10:
            # Fetch redditor data
            # Check if redditor data exist
            # redditor_dir = root_dir / 'data' / 'redditors'
            # if not redditor_dir.exists():
            #     redditor_dir.mkdir()
            # redditor_file = redditor_dir / f'{row["author_fullname"]}.csv'
            # if not redditor_file.exists():
            #     try:
            #         redditor = reddit.redditor(fullname=row['author_fullname'])
            #         redditorId = redditor.name  # Fetch redditor data
            #         redditor_dict = redditor.__dict__
            #         redditor_field = {key: redditor_dict[key] for key in redditor_dict.keys() &
            #                           {'id', 'name', 'icon_img', 'verified', 'created_utc'}}
            #         redditor_field['fullname'] = row['author_fullname']
            #         path = redditor_dir / f'{row["author_fullname"]}.csv'
            #         redditor_df = pandas.DataFrame.from_records([redditor_field])
            #         redditor_df.to_csv(path, index=False)
            #     except Exception as e:
            #         print(f'Redditor: {row["author_fullname"]} does not exist.')
            #         continue

            # Fetch comment by post's id
            try:
                submission = reddit.submission(row.id)
                top_level_comments = list(submission.comments)[:5]
                for comment in top_level_comments:
                    cId = comment.id
                    comment_field = dict(id=cId,
                                         name=comment.name,
                                         body=comment.body_html,
                                         score=comment.score,
                                         created_utc=comment.created_utc,
                                         redditId=row.id
                                         )
                    comment_list.append(comment_field)
            except Exception as e:
                print(f'Reddit id: {row.id} does not exist.')
                continue

    # Save the comment data
    path = reddit_data_dir / 'comment.csv'
    if len(comment_list) > 0:
        comment_df = pandas.DataFrame.from_records(comment_list)
        comment_df.to_csv(path, index=False)
    else:
        print('No comment!')
        pd.DataFrame().to_csv(path, index=False)


def fetch_reddits_social_context(
        max_results_per_query: int = 1_000,
        reddit_data_dir: str = 'data/reddit'):
    '''Extract keywords from all claims and fetch reddits for each of them.

    Args:
        max_results_per_query (int, optional):
            The maximal number of results to get per query. Must be at least
            10. Defaults to 1_000.
        reddit_data_dir (str, optional):
            The (relative or absolute) path to the directory containing all the
            tweets. Defaults to 'data/reddit'.
    '''

    # Initialise a praw.Reddit instance
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
        refresh_token=os.getenv('REDDIT_REFRESH_TOKEN'),
        user_agent=os.getenv('REDDIT_USER_AGENT'),
    )

    # Iterate over all the reddit data
    reddit_dir = root_dir / 'data' / 'reddit'
    desc = 'Fetching reddits\' social context'
    for query in tqdm(list(reddit_dir.iterdir()), desc=desc):
        reddit_data_dir = reddit_dir / query

        comment_file = reddit_data_dir / 'comment.csv'
        if comment_file.exists() and os.path.getsize(comment_file) > 1:
            continue

        post_file = glob.glob(f'{reddit_data_dir}/claim*.csv')
        if len(post_file) == 0:
            print(f'{reddit_data_dir} gets no claim file.')
            continue
        else:
            post_file = post_file[0]
        print(f'Processing file: {post_file}')

        # Load the CSV file with all the reddit posts relevant to the claim
        reddit_df = pd.read_csv(post_file, engine='python')

        fetch_social_context(reddit=reddit,
                             reddit_df=reddit_df,
                             reddit_data_dir=reddit_data_dir)
