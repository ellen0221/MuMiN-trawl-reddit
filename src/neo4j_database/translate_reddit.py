'''Translate all the tweets in the graph database'''

from tqdm.auto import tqdm
import time
from ..translator import GoogleTranslator
from .graph_reddit import GraphReddit


def translate_claims(batch_size: int, preprocess: bool = False):
    '''Translate all the claims in the graph database'''

    # Initialise graph and translator
    graph = GraphReddit()
    translator = GoogleTranslator()

    # Define query that counts how many claims there are left
    count_query = '''
        MATCH (c:Claim)
        WHERE c.claim_en IS NULL
        RETURN count(c) as num_claims
    '''

    preprocess_count_query = '''
        MATCH (c:Claim)
        WHERE c.claim_en IS NULL AND c.language = 'en'
        RETURN count(c) as num_claims
    '''

    # Define query that retrieves the claims
    get_query = f'''
        MATCH (c:Claim)
        WHERE c.claim_en IS NULL AND NOT (c.language = 'en')
        RETURN c.claim as claim
        LIMIT {batch_size}
    '''

    preprocess_get_query = f'''
        MATCH (c:Claim)
        WHERE c.claim_en IS NULL AND c.language = 'en'
        RETURN c.claim as claim
        LIMIT {batch_size}
    '''

    # Define query that sets the `claim_en` property of the claims
    set_query = '''
        UNWIND $claim_records as claimRecord
        MATCH (c:Claim {claim: claimRecord.claim})
        SET c.claim_en = claimRecord.claim_en
    '''

    # Preprocessing claims that are already in English
    if preprocess:
        desc = 'Preprocessing english claims ...'
        total_pre = graph.query(preprocess_count_query).num_claims[0]
        pbar_pre = tqdm(desc=desc, total=total_pre)
        while graph.query(preprocess_count_query).num_claims[0] > 0:
            claim_df = graph.query(preprocess_get_query)
            claim_df['claim_en'] = claim_df.claim
            claim_records = claim_df.to_dict('records')
            graph.query(set_query, claim_records=claim_records)
            pbar_pre.update(len(claim_df))
        pbar_pre.close()

    total = graph.query(count_query).num_claims[0]
    pbar = tqdm(desc='Translating claims', total=total)
    while graph.query(count_query).num_claims[0] > 0:
        claim_df = graph.query(get_query)
        claim_df['claim_en'] = translator(claim_df.claim)
        claim_records = claim_df.to_dict('records')
        graph.query(set_query, claim_records=claim_records)
        pbar.update(len(claim_df))

    pbar.close()
