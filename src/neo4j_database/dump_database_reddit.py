'''Make a dump of the database, for public release'''

from pathlib import Path
import logging
import io
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from typing import Dict, Union, List
import warnings
import zipfile
from src.neo4j_database.graph_reddit import GraphReddit
from src.utils import UpdateableZipFile, root_dir


logger = logging.getLogger(__name__)


def dump_database_reddit(overwrite: bool = False,
                         dump_nodes: bool = True,
                         dump_relations: bool = True):
    '''Dumps the database for public release.

    Args:
        overwrite (bool, optional):
            Whether to overwrite all files if they already exist. Defaults to
            False.
        dump_nodes (bool, optional):
            Whether to dump nodes in the graph. Defaults to True.
        dump_relations (bool, optional):
            Whether to dump relations in the graph. Defaults to True.
    '''
    # Remove database file if `overwrite` is True
    if overwrite:
        zip_file = root_dir / 'data' / 'db_dump' / 'mumin_reddit.zip'
        if zip_file.exists():
            zip_file.unlink()

    # Define node queries
    claim_query = '''
        MATCH (rev:Reviewer)-[r:HAS_REVIEWED]->(n:Claim)-[:HAS_LABEL]->(l:Label)
        WHERE r.predicted_verdict IN ['misinformation', 'factual']
        WITH DISTINCT n,
             collect(DISTINCT rev.url) AS reviewers,
             collect(l.verdict)[0] AS label
        RETURN id(n) AS id,
               label,
               reviewers
    '''
    reddit_query = '''
        MATCH (n:Reddit)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        RETURN DISTINCT n.redditId AS reddit_id, r.score AS relevance
    '''
    comment_query = '''
        MATCH (n:Reddit)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT n, r.score AS relevance
        MATCH (n)-[:HAS_COMMENT]->(c:Comment)
        RETURN DISTINCT c.commentId AS comment_id, relevance
    '''
    user_query = '''
        MATCH (re:Reddit)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH re, r.score AS relevance
        MATCH (u:User)-[:POSTED]->(re)
        RETURN DISTINCT u.userFullname AS user_id, relevance
    '''

    # Define relation queries
    user_posted_reddit_query = '''
        MATCH (tgt:Reddit)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT tgt, r.score AS relevance
        MATCH (src:User)-[:POSTED]->(tgt)
        RETURN src.userFullname AS src, tgt.redditId AS tgt, relevance
    '''
    reddit_discusses_claim_query = '''
        MATCH (src:Reddit)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(tgt:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        RETURN DISTINCT src.redditId AS src, id(tgt) AS tgt, r.score AS relevance
    '''
    reddit_has_comment_query = '''
        MATCH (src:Reddit)-[r:SIMILAR|SIMILAR_VIA_ARTICLE]->(tgt:Claim)-[:HAS_LABEL]->(:Label)
        WHERE r.score > 0.7
        WITH DISTINCT src, r.score AS relevance
        MATCH (src)-[:HAS_COMMENT]->(c:Comment)
        RETURN src.redditId AS src, c.commentId AS tgt, relevance
    '''

    # Define data types
    claim_types = dict(label='category',
                       reviewers='str')
    reddit_types = dict(reddit_id='str')
    comment_types = dict(comment_id='str')
    user_types = dict(user_id='str')

    # Dump nodes
    nodes = dict(reddit=(reddit_query, reddit_types),
                 claim=(claim_query, claim_types),
                 comment=(comment_query, comment_types),
                 user=(user_query, user_types))
    if dump_nodes:
        for name, (query, types) in nodes.items():
            dump_node(name, query, types)

    # Dump relations
    relations = dict(user_posted_reddit=user_posted_reddit_query,
                     reddit_discusses_claim=reddit_discusses_claim_query,
                     reddit_has_comment=reddit_has_comment_query)
    if dump_relations:
        for name, query in relations.items():
            dump_relation(name, query)


def dump_node(name: str,
              node_query: Union[str, List[str]],
              data_types: Dict[str, str]):
    '''Dumps a node to a file.

    The file will be saved to `/data/db_dump/mumin_reddit.zip`.

    Args:
        name (str):
            The name of the node, which will be the file name of the dump.
        node_query (str or list of str):
            The Cypher query that returns the node in question. Can also be
            list of such queries, in which case the deduplicated concatenated
            results from all of them are dumped. Each query should return the
            same columns.
        data_types (dict):
            A dictionary with keys all the columns of the resulting dataframe
            and values the datatypes. Here 'numpy' is reserved for converting
            the given column into a numpy ndarray.
    '''
    # Initialise paths
    dump_dir = root_dir / 'data' / 'db_dump'

    # Define `node_queries`, a list of queries
    if isinstance(node_query, str):
        node_queries = [node_query]
    else:
        node_queries = node_query

    # Ensure that the `data/db_dump` directory exists
    if not dump_dir.exists():
        dump_dir.mkdir()

    # Initialise graph database connection
    graph = GraphReddit()

    # Get the data from the graph database, as a Pandas DataFrame object
    logger.info(f'Fetching node data on "{name}"')
    df = pd.concat([graph.query(q) for q in node_queries], axis=0)

    if len(df):

        # Replace the relevance with the maximal relevance, in case of
        # duplicates
        if 'relevance' in df.columns:
            maxs = (df[[df.columns[0], 'relevance']]
                    .groupby(by=df.columns[0])
                    .max())
            df.relevance = (df.iloc[:, 0]
                              .map(lambda idx: maxs.relevance.loc[idx]))

        # Drop duplicate rows
        try:
            df.drop_duplicates(subset=df.columns[0], inplace=True)
        except TypeError as e:
            logger.info(f'Dropping of duplicates failed. The error was {e}.')

        # Convert the data types in the dataframe
        data_types_no_numpy = {col: typ for col, typ in data_types.items()
                               if typ != 'numpy'}
        df = df.astype(data_types_no_numpy)
        for col, typ in data_types.items():
            if typ == 'numpy':
                df[col] = df[col].map(lambda x: np.asarray(x))

        # Save the dataframe
        logger.info(f'Fetched {len(df)} {name} nodes. Saving node data')
        file_name = name + '_nodes'
        path = dump_dir / f'{file_name}.zip'
        compression_options = dict(method='zip', archive_name=f'{file_name}.csv')
        df.to_csv(path, compression=compression_options)


def dump_relation(name: str, relation_query: str):
    '''Dumps a relation to a file.

    The dataframe will have a `src` column for the source node ID, a `tgt`
    column for the target node ID, such that two IDs on the same row are
    related by the relation . Further columns will be relation features. The
    file will be saved to `/data/db_dump/mumin_reddit.zip`.

    Args:
        name (str):
            The name of the relation, which will be the file name of the dump.
        relation_query (str):
            The Cypher query that returns the node in question. The result of
            the query must return a `src` column for the node ID of the source
            node and a `tgt` column for the node ID of the target column.

    Raises:
        RuntimeError:
            If the columns of the resulting dataframe do not contain `src` and
            `tgt`.
    '''
    # Initialise paths
    dump_dir = root_dir / 'data' / 'db_dump'
    path = dump_dir / 'mumin_reddit.zip'

    # Define `relation_queries`, a list of queries
    if isinstance(relation_query, str):
        relation_queries = [relation_query]
    else:
        relation_queries = relation_query

    # Ensure that the `data/dump` directory exists
    if not dump_dir.exists():
        dump_dir.mkdir()

    # Initialise graph database connection
    graph = GraphReddit()

    # Get the data from the graph database, as a Pandas DataFrame object
    logger.info(f'Fetching relation data on "{name}"')
    df = pd.concat([graph.query(q) for q in relation_queries], axis=0)

    if len(df):

        # Replace the relevance with the maximal relevance, in case of
        # duplicates
        if 'relevance' in df.columns:
            df['src_tgt'] = [f'{src}_{tgt}'
                             for src, tgt in zip(df.src, df.tgt)]
            maxs = (df[['src_tgt', 'relevance']]
                    .groupby(by='src_tgt')
                    .max())
            df.relevance = (df.src_tgt
                              .map(lambda idx: maxs.relevance.loc[idx]))
            df.drop(columns='src_tgt', inplace=True)

        # Drop duplicate rows
        try:
            df.drop_duplicates(subset=['src', 'tgt'], inplace=True)
        except TypeError as e:
            logger.info(f'Dropping of duplicates failed. The error was {e}.')

        # Raise exception if the columns are not exactly `src` and `tgt`
        if 'src' not in df.columns or 'tgt' not in df.columns:
            raise RuntimeError(f'The dataframe for the relation "{name}" '
                               f'does not contain the columns `src` and '
                               f'`tgt`.')

        # Save the dataframe
        logger.info(f'Fetched {len(df)} {name} relation. Saving relation data')
        file_name = name + '_relation'
        path = dump_dir / f'{file_name}.zip'
        compression_options = dict(method='zip', archive_name=f'{file_name}.csv')
        df.to_csv(path, compression=compression_options)
