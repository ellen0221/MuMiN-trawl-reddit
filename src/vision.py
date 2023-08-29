'''GoogleVision classes'''

import logging
from dotenv import load_dotenv; load_dotenv()
from google.cloud import vision


logger = logging.getLogger(__name__)


class GoogleVision:
    '''A wrapper for the Google Translation API'''
    base_url = 'https://vision.googleapis.com/v1/images:annotate'

    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def __call__(self,
                 content: bytes):
        '''Extract features from image

        Args:
            imgUrls (list of str): The urls of the images that need to be extracted features.

        Returns:
            list of dictionary: The features and the url of each image.
        '''
        return batch_extract_properties(content, self.client)


# Extracting image features via Google Vision API
def batch_extract_properties(content: bytes, client: vision.ImageAnnotatorClient):
    """Detects image properties in the file located in Google Cloud Storage or
    on the Web."""
    image = vision.Image(content=content)
    # image.source.image_uri = url

    response = client.image_properties(image=image)
    props = response.image_properties_annotation

    # Get the first dominant color
    dominant_color = props.dominant_colors.colors[0]
    color_dict = {"frac": dominant_color.pixel_fraction, "r": dominant_color.color.red,
                  "g": dominant_color.color.green, "b": dominant_color.color.blue}

    if response.error.message:
        print(f'Response Error: {response.error.message}')

    return color_dict
