"""
The pre-trained models produce embeddings of size 512 - 1024. However, when storing a large
number of embeddings, this requires quite a lot of memory / storage.

In this example, we reduce the dimensionality of the embeddings to e.g. 128 dimensions. This significantly
reduces the required memory / storage while maintaining nearly the same performance.

For dimensionality reduction, we compute embeddings for a large set of (representative) sentence. Then,
we use PCA to find e.g. 128 principle components of our vector space. This allows us to maintain
us much information as possible with only 128 dimensions.

PCA gives us a matrix that down-projects vectors to 128 dimensions. We use this matrix
and extend our original SentenceTransformer model with this linear downproject. Hence,
the new SentenceTransformer model will produce directly embeddings with 128 dimensions
without further changes needed.
"""
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer, LoggingHandler, util, evaluation, models, InputExample
import numpy as np
import torch
from src.neo4j_database.graph_reddit import GraphReddit

get_query = '''
    MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(:Claim)
    WITH r
    MATCH (c:Comment)<-[:HAS_COMMENT]-(r:Reddit)
    WHERE c.parsed IS NULL
    RETURN c.text as sentence
    LIMIT 20000
'''

# Initialise the graph database
graph = GraphReddit()

# Model for which we apply dimensionality reduction
model = SentenceTransformer('all-MiniLM-L6-v2')

#New size for the embeddings
new_dimension = 50
######## Reduce the embedding dimensions ########

df = graph.query(get_query)

#To determine the PCA matrix, we need some example sentence embeddings.
#Here, we compute the embeddings for 20k comments from the graph dataset
pca_train_sentences = df.sentence.tolist()
train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True)

#Compute PCA on the train embeddings matrix
pca = PCA(n_components=new_dimension)
pca.fit(train_embeddings)
pca_comp = np.asarray(pca.components_)

# We add a dense layer to the model, so that it will produce directly embeddings with the new size
dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
model.add_module('dense', dense)

test_embedding = model.encode([pca_train_sentences[0]])
print(test_embedding.shape)

model.save_to_hub(repo_name='ellen-0221/50dim-all-MiniLM-L6-v2', private=True, exist_ok=True)

# You can then load the adapted model that produces 128 dimensional embeddings like this:
#model = SentenceTransformer('models/my-128dim-model')
