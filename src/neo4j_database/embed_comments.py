'''Embed the related comments in the graph database'''

from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd
from typing import Callable
from .graph_reddit import GraphReddit


def embed(get_query: str,
          set_embedding_query: str,
          total_count_query: str,
          remaining_count_query: str,
          start_from_scratch_query: str,
          doc_preprocessing_fn: Callable = lambda x: x,
          start_from_scratch: bool = False,
          gpu: bool = False):
    '''Abstract function used to embed various nodes in the graph database.

    Args:
        get_query (str):
            The cypher query used to get a new batch. Has to output two
            columns, 'id' and 'doc', with the 'id' uniquely identifying the
            given node, and 'doc' the string to be embedded.
        set_embedding_query (str):
            The cypher query used to set the retrieved embeddings. Must take
            'id_embeddings' as an argument, which is a list of dicts, each of
            which has an 'id' entry and an 'embedding' entry.
        total_count_query (str):
            The cypher query used to count the total number of relevant nodes.
            The column containing the count must be named 'num'.
        remaining_count_query: (str):
            The cypher query used to count the remaining nodes that have not
            been embedded yet. The column containing the count must be named
            'num'.
        start_from_scratch_query (str, optional):
            The cypher query used to remove all the relevant embeddings from
            the nodes.
        doc_preprocessing_fn (callable, optional):
            A function that preprocesses the documents fetched by `get_query`.
            Needs to be a function that inputs and outputs a Pandas DataFrame,
            both of which have columns 'id' and 'doc'. Defaults to the identity
            function.
        start_from_scratch (bool, optional):
            Whether to start from scratch or not. Defaults to False.
        gpu (bool, optional):
            Whether to use the GPU. Defaults to True.
    '''
    # Initialise the graph database
    graph = GraphReddit()

    # Remove all embeddings from the graph
    if start_from_scratch:
        graph.query(start_from_scratch_query)

    # Load the embedding model and its tokenizer
    # device = 'cuda' if gpu else 'cpu'
    # model = SentenceTransformer('models/50dim-all-MiniLM-L6-v2', device=device)
    tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': 512}
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                          top_k=1)

    # Get the total number of comments and define a progress bar
    total = graph.query(total_count_query).num[0]
    pbar = tqdm(total=total, desc='Adding comment embeddings and emotion')
    num_embedded = graph.query(remaining_count_query).num[0]
    pbar.update(total - num_embedded)

    # Continue embedding until every node has been embedded
    while graph.query(remaining_count_query).num[0] > 0:
        # Fetch new ids and docs
        df = graph.query(get_query)
        df = doc_preprocessing_fn(df)
        ids = df.id.tolist()
        docs = df.doc.tolist()


        # Embed the docs
        # embeddings = [list(emb.astype(float)) for emb in model.encode(docs)]
        embeddings = [result[0] for result in classifier(docs, **tokenizer_kwargs)]

        # Set the summary as an attribute on the nodes
        id_embeddings = [dict(id=id, embedding=embedding)
                         for id, embedding in zip(ids, embeddings)]
        graph.query(set_embedding_query, id_embeddings=id_embeddings)

        # Update the progress bar
        pbar.update(len(docs))
        pbar.total = graph.query(total_count_query).num[0]
        pbar.refresh()

    # Close the progress bar
    pbar.close()


def embed_comments(start_from_scratch: bool = False, gpu: bool = False):
    '''Embed all the labelled Reddit posts' comments in the graph database.

    Args:
        start_from_scratch (bool, optional):
            Whether to initially remove all the embeddings. Defaults to
            False.
        gpu (bool, optional):
            Whether to use the GPU. Defaults to False.
    '''
    get_query = '''
        MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(:Claim)
        WITH r
        MATCH (c:Comment)<-[:HAS_COMMENT]-(r:Reddit)
        WHERE c.embedding IS NULL
        RETURN c.text as doc, id(c) as id
        LIMIT 500
    '''
    set_embedding_query = '''
        UNWIND $id_embeddings AS id_embedding
        WITH id_embedding.id AS id,
             id_embedding.embedding AS embedding
        MATCH (n:Comment)
        WHERE id(n)=id
        SET n.embedding = embedding
    '''
    total_count_query = '''
        MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(:Claim)
        WITH r
        MATCH (c:Comment)<-[:HAS_COMMENT]-(r:Reddit)
        RETURN count(c) AS num
    '''
    remaining_count_query = '''
        MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(:Claim)
        WITH r
        MATCH (c:Comment)<-[:HAS_COMMENT]-(r:Reddit)
        WHERE c.embedding IS NULL
        RETURN count(c) AS num
    '''
    start_from_scratch_query = '''
        MATCH (n:Comment)
        WHERE n.embedding IS NOT NULL
        REMOVE n.embedding
    '''

    def clean_comment(df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()
        cleaned_df['doc'] = cleaned_df.doc.str.lower().str.strip()
        return cleaned_df

    embed(get_query=get_query,
          set_embedding_query=set_embedding_query,
          total_count_query=total_count_query,
          remaining_count_query=remaining_count_query,
          start_from_scratch_query=start_from_scratch_query,
          start_from_scratch=start_from_scratch,
          doc_preprocessing_fn=clean_comment,
          gpu=gpu)

def extract_comments_emotion(start_from_scratch: bool = False, gpu: bool = False):
    '''Embed all the labelled Reddit posts' comments in the graph database.

    Args:
        start_from_scratch (bool, optional):
            Whether to initially remove all the embeddings. Defaults to
            False.
        gpu (bool, optional):
            Whether to use the GPU. Defaults to False.
    '''
    get_query = '''
        MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(:Claim)
        WITH r
        MATCH (c:Comment)<-[:HAS_COMMENT]-(r:Reddit)
        WHERE c.emotion IS NULL
        RETURN c.text as doc, id(c) as id
        LIMIT 500
    '''
    set_embedding_query = '''
        UNWIND $id_embeddings AS id_embedding
        WITH id_embedding.id AS id,
             id_embedding.embedding AS embedding
        MATCH (n:Comment)
        WHERE id(n)=id
        SET n.emotion = embedding.label, n.emotion_score = embedding.score
    '''
    total_count_query = '''
        MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(:Claim)
        WITH r
        MATCH (c:Comment)<-[:HAS_COMMENT]-(r:Reddit)
        RETURN count(c) AS num
    '''
    remaining_count_query = '''
        MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(:Claim)
        WITH r
        MATCH (c:Comment)<-[:HAS_COMMENT]-(r:Reddit)
        WHERE c.emotion IS NULL
        RETURN count(c) AS num
    '''
    start_from_scratch_query = '''
        MATCH (n:Comment)
        WHERE n.emotion IS NOT NULL
        REMOVE n.emotion
    '''

    def clean_comment(df: pd.DataFrame) -> pd.DataFrame:
        cleaned_df = df.copy()
        cleaned_df['doc'] = cleaned_df.doc.str.lower().str.strip()
        return cleaned_df

    embed(get_query=get_query,
          set_embedding_query=set_embedding_query,
          total_count_query=total_count_query,
          remaining_count_query=remaining_count_query,
          start_from_scratch_query=start_from_scratch_query,
          start_from_scratch=start_from_scratch,
          doc_preprocessing_fn=clean_comment,
          gpu=gpu)