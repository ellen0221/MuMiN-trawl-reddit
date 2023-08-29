import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger('urllib3.connection').setLevel(logging.CRITICAL)
logging.getLogger('jieba').setLevel(logging.CRITICAL)


def train():
    from src.verdict_classifier.train import train
    train()


def create_claims(start_from_scratch: bool = False):
    from src.neo4j_database.populate import create_claims
    create_claims(start_from_scratch=start_from_scratch)


def create_claims_reddit(start_from_scratch: bool = False):
    from src.neo4j_database.populate_reddit import create_claims
    create_claims(start_from_scratch=start_from_scratch)


def create_articles(start_from_scratch: bool = False,
                    num_workers: int = 14,
                    skip_preparse: bool = True):
    from src.neo4j_database.populate import create_articles
    create_articles(start_from_scratch=start_from_scratch,
                    num_workers=num_workers,
                    skip_preparse=skip_preparse)


def create_articles_reddit(start_from_scratch: bool = False,
                           num_workers: int = 14,
                           skip_preparse: bool = True):
    from src.neo4j_database.populate_reddit import create_articles
    create_articles(start_from_scratch=start_from_scratch,
                    num_workers=num_workers,
                    skip_preparse=skip_preparse)


def summarise_articles(start_from_scratch: bool = False):
    from src.neo4j_database.summarise import summarise_articles
    summarise_articles(start_from_scratch=start_from_scratch)


def summarise_articles_reddit(start_from_scratch: bool = False):
    from src.neo4j_database.summarise_reddit import summarise_articles
    summarise_articles(start_from_scratch=start_from_scratch)


def embed_all(start_from_scratch: bool = False):
    from src.neo4j_database.embed import embed_articles, embed_claims, embed_tweets
    embed_claims(start_from_scratch=start_from_scratch, gpu=True)
    embed_tweets(start_from_scratch=start_from_scratch, gpu=True)
    embed_articles(start_from_scratch=start_from_scratch, gpu=True)


def embed_all_reddit(start_from_scratch: bool = False):
    from src.neo4j_database.embed_reddit import embed_articles, embed_claims, embed_reddits
    # embed_claims(start_from_scratch=start_from_scratch, gpu=False)
    # embed_reddits(start_from_scratch=start_from_scratch, gpu=False)
    embed_articles(start_from_scratch=start_from_scratch, gpu=False)


def populate(start_from_scratch: bool = False):
    from src.neo4j_database.populate import populate
    populate(start_from_scratch=start_from_scratch)


def populate_reddit(start_from_scratch: bool = False):
    from src.neo4j_database.populate_reddit import populate
    populate(start_from_scratch=start_from_scratch)


def fetch_tweets_from_extracted_keywords():
    from src.fetch_tweets import fetch_tweets_from_extracted_keywords
    fetch_tweets_from_extracted_keywords(
        max_results_per_query=100,
        gpu=False,
        twitter_data_dir='/media/secure/dan/twitter'
    )


def fetch_reddits_from_claims():
    from src.fetch_reddits import fetch_reddits_from_claims
    fetch_reddits_from_claims(
        max_results_per_query=100,
        gpu=False,
        # reddit_data_dir='/media/secure/dan/reddit'0
    )


def fetch_reddits_social_context():
    from src.fetch_reddits_social_context import fetch_reddits_social_context
    fetch_reddits_social_context()


def fetch_facts(num_results_per_reviewer: int = 100_000,
                translate: bool = True):
    from src.fetch_facts import fetch_facts
    fetch_facts(num_results_per_reviewer=num_results_per_reviewer,
                translate=translate)


def evaluate_verdict_classifier():
    from pathlib import Path
    from src.verdict_classifier import VerdictClassifier
    try:
        ckpt = next(Path('models').glob('*.ckpt'))
        transformer = 'sentence-transformers/paraphrase-mpnet-base-v2'
        model = VerdictClassifier.load_from_checkpoint(str(ckpt),
                                                       transformer=transformer)
        model.eval()
        verdicts = ['true', 'false', 'not sure', 'sure', 'half true',
                    'half false', 'mostly true', 'mostly false']
        print(model.predict(verdicts))
    except StopIteration:
        raise RuntimeError('No pretrained verdict classifier in `models`'
                           'directory!')


def add_predicted_verdicts():
    from src.add_predicted_verdicts import add_predicted_verdicts
    add_predicted_verdicts()


def link_tweets(start_from_scratch: bool = False):
    from src.neo4j_database.link_nodes import link_nodes
    link_nodes(start_from_scratch=start_from_scratch, node_type='tweet')


def link_articles(start_from_scratch: bool = False):
    from src.neo4j_database.link_nodes import link_nodes
    link_nodes(start_from_scratch=start_from_scratch, node_type='article')


def link_all(start_from_scratch: bool = False):
    from src.neo4j_database.link_nodes import link_nodes
    link_nodes(start_from_scratch=start_from_scratch, node_type='article')
    link_nodes(start_from_scratch=start_from_scratch, node_type='tweet')


def link_all_reddit(start_from_scratch: bool = False):
    from src.neo4j_database.link_nodes_reddit import link_nodes
    link_nodes(start_from_scratch=start_from_scratch, node_type='article', reset_claim_status=False)
    link_nodes(start_from_scratch=start_from_scratch, node_type='reddit', reset_claim_status=True)


def add_label_nodes(start_from_scratch: bool = False):
    from src.neo4j_database.link_nodes_reddit import add_label_nodes
    add_label_nodes(start_from_scratch=start_from_scratch, process_multiple_linked_nodes=False)


def translate_tweets():
    from src.neo4j_database.translate import translate_tweets
    translate_tweets(batch_size=1000)


def translate_claims():
    from src.neo4j_database.translate import translate_claims
    translate_claims(batch_size=1000)


def translate_claims_reddit():
    from src.neo4j_database.translate_reddit import translate_claims
    translate_claims(batch_size=1000, preprocess=True)


def translate_articles():
    from src.neo4j_database.translate import translate_articles
    translate_articles(batch_size=2)


def dump_database(overwrite: bool = False):
    from src.neo4j_database.dump_database import dump_database
    dump_database(overwrite=overwrite)


def dump_claim_embeddings(overwrite: bool = False):
    from src.neo4j_database.dump_database import dump_node
    query = '''
        MATCH (n:Claim)<-[r:SIMILAR]-()
        WITH DISTINCT n as n
        RETURN id(n) as id,
               n.claim_en as claim,
               n.embedding as embedding
    '''
    dump_node('claim_embeddings', query, overwrite=overwrite)


def dump_cosine_similarities(overwrite: bool = False):
    from src.neo4j_database.dump_database import dump_cosine_similarities
    dump_cosine_similarities()


def extract_image_features():
    from src.neo4j_database.image_feature_extraction import extract_image_features
    extract_image_features()

def dump_databse_reddit(overwrite: bool = False):
    from src.neo4j_database.dump_database_reddit import dump_database_reddit
    dump_database_reddit()


if __name__ == '__main__':
    # dump_database(overwrite=True)
    # link_all()
    # embed_all()
    # dump_cosine_similarities()
    # fetch_facts()
    # add_predicted_verdicts()
    # fetch_reddits_from_claims()
    # fetch_reddits_social_context()
    # create_claims_reddit()
    # populate_reddit()
    # translate_claims_reddit()
    # create_articles_reddit()
    # summarise_articles_reddit()
    # embed_all_reddit()
    # link_all_reddit()
    # extract_image_features()
    dump_databse_reddit()
