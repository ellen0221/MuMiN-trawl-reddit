'''Populate the graph database'''

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import datetime as dt
from typing import List, Optional
from newspaper import Article, ArticleException
from tqdm.auto import tqdm
import json
import re
from neo4j.exceptions import DatabaseError
from concurrent.futures import ProcessPoolExecutor
from timeout_decorator import timeout, TimeoutError
import glob
import os

from ..utils import root_dir, strip_url
from .graph_reddit import GraphReddit


def populate(start_from_scratch: bool = False,
             queries: Optional[List[str]] = None):
    '''Populate the graph database with nodes and relations'''
    # Initialise the graph database
    graph = GraphReddit()

    # Set up Reddit posts directory and a list of all the Reddit queryies
    reddit_dir = root_dir / 'data' / 'reddit'
    if queries is None:
        queries = [p for p in reddit_dir.iterdir()]

    # Delete all nodes and constraints in database
    if start_from_scratch:
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH (n) RETURN n",'
                    '"DETACH DELETE n",'
                    '{batchsize:10000, parallel:false})')

        constraints = graph.query('CALL db.constraints')
        for constraint in constraints.name:
            graph.query(f'DROP CONSTRAINT {constraint}')

    # Set up cypher directory
    cypher_dir = root_dir / 'src' / 'neo4j_database' / 'cypher_reddit'
    constraint_paths = list(cypher_dir.glob('constraint_*.cql'))
    node_paths = list(cypher_dir.glob('node_*.cql'))
    rel_paths = list(cypher_dir.glob('rel_*.cql'))

    # Create constraints
    for path in tqdm(constraint_paths, desc='Creating constraints'):
        cypher = path.read_text()
        graph.query(cypher)

    def load_records(file_path: str) -> pd.DataFrame:
        '''Helper function to load records from a CSV file'''
        try:
            df = pd.read_csv(file_path, engine='python')
            df = df.replace({np.nan: None})
            return df
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def try_int(x):
        '''Check whether a string could be converted to int'''
        try:
            int(x)
            return True
        except ValueError:
            return False

    def filter_dataframes(dataframes: dict, min_comments: int = 10) -> dict:
        '''Filter the dataframes to only contain the popular Reddit posts'''
        # Filter the reddit dataframe
        reddits = dataframes['reddits']
        # Filter out missing values
        reddits = reddits.dropna(subset=['num_comments'])
        if not reddits.empty:
            # Make sure that all values in the num_comments column can be converted to int
            reddits = reddits[[try_int(x) for x in reddits['num_comments']]]
            reddits = reddits[reddits['num_comments'].astype('int') >= min_comments]
        if not reddits.empty and 'post_hint' in reddits.columns:
            # Filter out rows with the value of the 'post_hint' column contains 'video'
            reddits = reddits[~(reddits['post_hint'].astype(str).str.contains("video"))]
        dataframes['reddits'] = reddits

        return dataframes

    # Create nodes and relations, one query at a time
    total = (len(node_paths) + len(rel_paths)) * len(queries)
    with tqdm(total=total) as pbar:
        for query in queries:
            reddit_data_dir = reddit_dir / query

            post_file = glob.glob(f'{reddit_data_dir}/claim*.csv')
            if len(post_file) == 0:
                print(f'{reddit_data_dir} gets no claim file.')
                continue
            else:
                post_file = post_file[0]
            dfs = dict(reddits=load_records(post_file))
            comment_file = reddit_data_dir / 'comment.csv'
            if comment_file.exists() and os.path.getsize(comment_file) > 1:
                dfs['comments'] = load_records(str(comment_file))
            else:
                dfs['comments'] = pd.DataFrame()

            # Filter out all the Reddit data not concerning a popular Reddit post
            dfs = filter_dataframes(dfs)
            records = {key: df.to_dict('records') for key, df in dfs.items()}

            # Create all the nodes
            for path in node_paths:
                desc = f'Creating nodes ({query} - {path.stem})'
                pbar.set_description(desc)
                cypher = path.read_text()
                try:
                    graph.query(cypher, **records)
                except Exception as e:
                    print(f'The query "{query}" failed when trying to create '
                          f'the node {path.stem}. The specific error is: {e}.')
                pbar.update(1)

            # Create all the relations
            for path in rel_paths:
                desc = f'Creating relations ({query} - {path.stem})'
                pbar.set_description(desc)
                cypher = path.read_text()
                try:
                    graph.query(cypher, **records)
                except Exception as e:
                    print(f'The query "{query}" failed when trying to create '
                          f'the relation {path.stem}. The specific error is: '
                          f'{e}.')
                pbar.update(1)


@timeout(5)
def download_article_with_timeout(article):
    article.download()
    return article


def process_url(url: str):
    '''Helper function to process a url, used when creating Article nodes.'''

    # Strip the url of GET arguments
    stripped_url = strip_url(url)

    # Initialise the graph database
    graph = GraphReddit()

    # Define the query which tags the URL as parsed
    set_parsed_query = '''
        MATCH (url:Url {name:$url})
        SET url.parsed = true
    '''

    # Check if the Article node already exists and skip if it is the case
    query = '''
        MATCH (n:Article {url:$url})
        RETURN count(n) as numArticles
    '''
    if graph.query(query, url=stripped_url).numArticles[0] > 0:
        graph.query(set_parsed_query, url=url)
        return None

    # Initialise article
    try:
        article = Article(stripped_url)
        article = download_article_with_timeout(article)
        article.parse()
    except (ArticleException, ValueError, RuntimeError, TimeoutError):
        graph.query(set_parsed_query, url=url)
        return None

    # Extract the title and skip URL if it is empty
    title = article.title
    if title == '':
        graph.query(set_parsed_query, url=url)
        return None
    title = re.sub('\n+', '\n', title)
    title = re.sub(' +', ' ', title)
    title = title.strip()

    # Extract the content and skip URL if it is empty
    content = article.text.strip()
    if content == '':
        graph.query(set_parsed_query, url=url)
        return None
    content = re.sub('\n+', '\n', content)
    content = re.sub(' +', ' ', content)
    content = content.strip()

    # Extract the authors, the publishing date and the top image
    authors = article.authors
    if article.publish_date is not None:
        publish_date = dt.datetime.strftime(article.publish_date, '%Y-%m-%d')
    else:
        publish_date = None
    top_image = article.top_image

    try:
        # Create the corresponding Article node
        query = '''
            MERGE (n:Article {url:$url})
            SET
                n.title = $title,
                n.authors = $authors,
                n.publish_date = $publish_date,
                n.content = $content
        '''
        graph.query(query, url=stripped_url, title=title, authors=authors,
                    publish_date=publish_date, content=content)

        # Link the Article node to the Url node
        query = '''
            MATCH (url:Url {name:$url})
            MATCH (article:Article {url:$stripped_url})
            MERGE (url)-[:IS_ARTICLE]->(article)
        '''
        graph.query(query, url=url, stripped_url=stripped_url)

        # Create the Image node for the top image if it doesn't already
        # exist
        if top_image is not None and top_image.strip() != '':
            # Create an `Image` node for the top image
            query = '''
                MERGE (:Image {url:$top_image})
            '''
            graph.query(query, top_image=top_image)

            # Connect the Article node with the top image Url node
            query = '''
                MATCH (image:Image {url:$top_image})
                MATCH (article:Article {url:$url})
                MERGE (article)-[:HAS_TOP_IMAGE]->(image)
            '''
            graph.query(query, top_image=top_image, url=stripped_url)

    # If a database error occurs, such as if the URL is too long, then
    # skip this URL
    except DatabaseError:
        graph.query(set_parsed_query, url=url)
        return None

    # Tag the Url node as parsed
    graph.query(set_parsed_query, url=url)


def create_articles(start_from_scratch: bool = False,
                    num_workers: int = 8,
                    skip_preparse: bool = True):
    '''Create article nodes from the URL nodes present in the graph database'''
    # Initialise graph
    graph = GraphReddit()

    if start_from_scratch:
        # Remove all articles
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH (n:Article) RETURN n",'
                    '"DETACH DELETE n",'
                    '{batchsize:10000, parallel:false})')

        # Remove `parsed` from all `Url` nodes
        query = '''MATCH (url: Url)
                   WHERE url.parsed IS NOT NULL
                   REMOVE url.parsed'''
        graph.query(query)

    # Mark certain URLs as `parsed` that are not articles
    if not skip_preparse:
        get_query = '''MATCH (url: Url)
                       WHERE url.parsed IS NULL AND
                             (url.name =~ '.*youtu[.]*be.*' OR
                              url.name =~ '.*vimeo.*' OR
                              url.name =~ '.*spotify.*' OR
                              url.name =~ '.*twitter.*' OR
                              url.name =~ '.*instagram.*' OR
                              url.name =~ '.*facebook.*' OR
                              url.name =~ '.*tiktok.*' OR
                              url.name =~ '.*redd\.it.*' OR
                              url.name =~ '.*reddit.*' OR
                              url.name =~ '.*gab[.]com.*' OR
                              url.name =~ 'https://t[.]me.*' OR
                              url.name =~ '.*imgur.*' OR
                              url.name =~ '.*/photo/.*' OR
                              url.name =~ '.*mp4' OR
                              url.name =~ '.*mov' OR
                              url.name =~ '.*jpg' OR
                              url.name =~ '.*jpeg' OR
                              url.name =~ '.*bmp' OR
                              url.name =~ '.*png' OR
                              url.name =~ '.*gif' OR
                              url.name =~ '.*pdf')
                       RETURN url'''
        set_query = 'SET url.parsed = true'
        query = (f'Call apoc.periodic.iterate("{get_query}", "{set_query}", '
                 f'{{batchsize:1000, parallel:false}})')
        graph.query(query)

    # Get total number of Url nodes
    query = '''MATCH (url: Url)
               WHERE url.parsed IS NULL
               RETURN count(url) as numUrls'''
    num_urls = graph.query(query).numUrls[0]

    # Get batch of urls and create progress bar
    if num_urls > 0:
        query = f'''MATCH (url: Url)
                    WHERE url.parsed IS NULL
                    RETURN url.name AS url
                    LIMIT {10 * num_workers}'''
        urls = graph.query(query).url.tolist()
        pbar = tqdm(total=num_urls, desc='Parsing Url nodes')
    else:
        # urls = list()
        return

    # Continue looping until all url nodes have been visited
    while True:

        if num_workers == 1:
            for url in urls:
                process_url(url)
                pbar.update()
        else:
            with ProcessPoolExecutor(num_workers) as pool:
                pool.map(process_url, urls, timeout=10)
            pbar.update(len(urls))

        # Get new batch of urls
        query = f'''MATCH (url: Url)
                    WHERE url.parsed IS NULL
                    RETURN url.name AS url
                    LIMIT {10 * num_workers}'''
        result = graph.query(query)
        if len(result) > 0:
            urls = graph.query(query).url.tolist()
        else:
            break

    # Close progress bar
    if num_urls > 0:
        pbar.close()


def create_claims(start_from_scratch: bool = False):
    '''Create claim nodes in the graph database'''
    # Initialise graph
    graph = GraphReddit()

    # Remove all claims
    if start_from_scratch:
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH (n:Claim) RETURN n",'
                    '"DETACH DELETE n",'
                    '{batchsize:10000, parallel:false})')
        graph.query('CALL apoc.periodic.iterate('
                    '"MATCH (n:Reviewer) RETURN n",'
                    '"DETACH DELETE n",'
                    '{batchsize:10000, parallel:false})')

    # Create the cypher query that creates a `Claim` node
    create_claim_query = '''
        MERGE (c:Claim {claim: $claim})
        SET
            c.source = $source,
            c.date = $date,
            c.language = $language
    '''

    create_reviewer_query = 'MERGE (n:Reviewer {url: $reviewer})'

    link_reviewer_claim_query = '''
        MATCH (n:Reviewer {url: $reviewer})
        MATCH (c:Claim {claim: $claim})
        MERGE (n)-[r:HAS_REVIEWED]->(c)
        SET
            r.raw_verdict = $raw_verdict,
            r.raw_verdict_en = $raw_verdict_en,
            r.predicted_verdict = $predicted_verdict
    '''

    link_reddit_claim_query = '''
        UNWIND $reddit_ids as reddit
        WITH reddit
        MATCH (n:Reddit {redditId: reddit})
        MATCH (c:Claim {claim: $claim})
        MERGE (n)-[r:SEARCH_FROM]->(c)
    '''

    reviewer_dir = Path('data') / 'reviewers'
    desc = 'Creating Claim nodes'
    for reviewer_csv in tqdm(list(reviewer_dir.iterdir()), desc=desc):
        graph.query(create_reviewer_query, reviewer=reviewer_csv.stem)
        reviewer_df = pd.read_csv(reviewer_csv)
        reddits_dir = root_dir / 'data' / 'reddit'
        for index, row in reviewer_df.iterrows():
            graph.query(create_claim_query,
                        claim=row.claim,
                        date=row.date,
                        source=row.source,
                        language=row.language)
            graph.query(link_reviewer_claim_query,
                        reviewer=row.reviewer,
                        claim=row.claim,
                        raw_verdict=row.raw_verdict,
                        raw_verdict_en=row.raw_verdict_en,
                        predicted_verdict=row.predicted_verdict)
            # Create the relationship 'search_from' between claims and reddit posts
            reddits_path = str(reddits_dir) + f'/{reviewer_csv.stem}-claim{int(index)}/claim{int(index)}.csv'
            try:
                reddits_df = pd.read_csv(reddits_path)
                reddit_ids = reddits_df['id'].values.tolist()
                graph.query(link_reddit_claim_query,
                            claim=row.claim,
                            reddit_ids=reddit_ids)
            except FileNotFoundError:
                print(f'No such file: {reddits_path}')
                continue
