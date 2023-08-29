from urllib.error import HTTPError

from tqdm.auto import tqdm
import io
import urllib.request
from PIL import Image
from keras.preprocessing import image
from ..vision import GoogleVision
from .graph_reddit import GraphReddit

get_reddit_images_query = '''
    MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(:Claim)
    WITH r
    MATCH (i:Image)<-[:HAS_IMAGE]-(r:Reddit)
    WHERE i.resizedArr IS NULL
    RETURN DISTINCT i.url as urls, id(i) as node_ids
'''

get_article_images_query = '''
    MATCH (r:Reddit)-[:SIMILAR|SIMILAR_VIA_ARTICLE]-(:Claim)
    WITH r
    MATCH (a:Article)<-[:HAS_ARTICLE]-(r:Reddit)
    WITH a
    MATCH (a)-[:HAS_TOP_IMAGE]-(i:Image)
    WHERE i.resizedArr IS NULL
    RETURN DISTINCT i.url as urls, id(i) as node_ids
'''

update_image_query = '''
    MATCH (i:Image)
    WHERE id(i)=$data.id AND i.parsed IS NULL
    SET i.width=$data.width, i.height=$data.height,
        i.size=$data.size, i.frac=$data.frac,
        i.r=$data.r, i.g=$data.g, i.b=$data.b,
        i.parsed=True
'''

set_parsed_query = '''
    MATCH (i:Image)
    WHERE id(i)=$idx
    SET i.parsed=True
'''


def extract_image_features():
    graph = GraphReddit()
    reddit_image_df = graph.query(get_reddit_images_query)
    if len(reddit_image_df) > 0:
        reddit_image_ids = reddit_image_df.node_ids.tolist()
        reddit_image_urls = reddit_image_df.urls.tolist()
        reddit_images = [dict(id=idx, url=url) for idx, url in zip(reddit_image_ids, reddit_image_urls)]
    else:
        reddit_images = []
    article_image_df = graph.query(get_article_images_query)
    article_image_ids = article_image_df.node_ids.tolist()
    article_image_urls = article_image_df.urls.tolist()
    article_images = [dict(id=idx, url=url) for idx, url in zip(article_image_ids, article_image_urls)]
    # Merge two list without duplicates
    image_urls = reddit_images + article_images
    for pic in tqdm(image_urls, desc='Processing images'):
        try:
            if '.gif' not in pic['url']:
                im = extract_image_size(pic['url'])
                image_byte = image_to_byte_array(im)
                image_properties = extract_dominant_color(image_byte)
                image_properties['width'] = im.width
                image_properties['height'] = im.height
                image_properties['size'] = im.width * im.height
                image_properties['id'] = pic['id']
                graph.query(update_image_query, data=image_properties)
            else:
                graph.query(set_parsed_query, idx=pic['id'])
        except HTTPError:
            print('Cannot access the image.')
            graph.query(set_parsed_query, idx=pic['id'])
            continue


def extract_image_size(url: str):
    # Get image's size, width, and height
    file = urllib.request.urlopen(url)
    im = Image.open(file)
    return im


def image_to_byte_array(img: Image) -> bytes:
    # BytesIO is a file-like buffer stored in memory
    imgByteArr = io.BytesIO()
    # image.save expects a file-like as a argument
    img.save(imgByteArr, format=img.format)
    # Turn the BytesIO object back into a bytes object
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr


def extract_dominant_color(content: bytes):
    # Extracting image features via Google Vision API
    vision = GoogleVision()
    image_prop = vision(content)
    return image_prop
