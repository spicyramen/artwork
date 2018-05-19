"""Use Google Cloud Vision API to extract labels from Movie images.

Using a list of images, locally, we use Google Vision API to recognize
labels in images, using these labels we will select the top _MAX_RESULTS labels
A result file is generated with Movie id and the associated labels as a bag of
words.
This script requires user to manually setup the Google Cloud credentials:

Please follow Cloud Quick start to set up credentials and enable API.
https://github.com/GoogleCloudPlatform/cloud-vision/tree/master/python

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
from collections import namedtuple
import csv
import glob
import io
import pandas
import ntpath

from absl import app
from absl import flags
from absl import logging

from apiclient import discovery
from concurrent import futures

FLAGS = flags.FLAGS

flags.DEFINE_string('api_key', None, 'API Key')
flags.DEFINE_string('folder', '', 'Folder where images are located.')
flags.DEFINE_string('results', '', 'File to store image labels.')
flags.DEFINE_boolean('graph', True, 'Analyze labels from results file.')

# Google Cloud information.
_API_NAME = 'vision'
_API_VERSION = 'v1'
_MAX_RESULTS = 10
_NUM_RETRIES = 3

# Image columns.
_MASTER_FIELD = 'master_id'
_FILE_LOCATION = 'path'
_LABELS = 'labels'

# File information.
_FILE_RESULTS = 'images_results.csv'

# Concurrent parameters.
_MAX_WORKERS = 10
_EXECUTOR_TIMEOUT = 60

# API information.
_RESPONSES = 'responses'
_ANNOTATIONS = 'labelAnnotations'
_DESCRIPTION = 'description'

ArtWorkImage = namedtuple('ArtWorkImage', (_MASTER_FIELD, _FILE_LOCATION))


def _GetService(api_key):
    """Gets service instance to start API searches.

    Args:
      api_key: key obtained from https://code.google.com/apis/console.

    Returns:
      A Google API Service used to send requests.
    """

    return discovery.build(_API_NAME, _API_VERSION, developerKey=api_key,
                           cache_discovery=False)


def _IsEmpty(data):
    """Check if Pandas dataFrame is empty or not.

    Args:
        data: (pandas.DataFrame) Dataframe to process.

    Returns:
        A boolean. True if the given data is interpreted as empty
    """

    if data is not None and not data.empty:
        return False
    return True


def LoadDataSet(folder):
    """Reads folder and generates a list of filenames path.

    Args:
      folder: (str) Local or remote readable path.

    Returns:
      A collections.Sequence with tuples of filename and paths.

    Raises:
      FileError: Unable to read filename.
      ValueError: Invalid folder, No data in file.
    """

    if not folder:
        raise ValueError('Invalid folder')

    logging.info('Reading folder: %s.', folder)
    image_list = glob.glob('%s/*.png' % folder)
    image_list.extend(glob.glob('%s/*.jpg' % folder))
    logging.info('Found %d images in folder: %s ', len(image_list), folder)
    return [(ntpath.basename(image), image) for image in image_list]


def LoadResultsFile(filename):
    """Reads filename and generates a Pandas DataFrame with labels.

    Args:
      filename: (str) Local CSV file used as input.

    Returns:
      Pandas.DataFrame: CSV contents.

    Raises:
      FileError: Unable to read filename.
      ValueError: Invalid file, No data in file.
    """

    if not filename:
        raise ValueError('Invalid filename')
    logging.info('Reading file: %s.', filename)
    # Generates a pandas.DataFrame from data in file.
    with open(filename) as input_file:
        dataframe = input_file.read()
    return pandas.read_csv(
        io.BytesIO(dataframe), names=[_MASTER_FIELD, _LABELS],
        header=None)


def SaveDataSet(image_list, filename):
    """Write list of lists in file.

    Args:
      movie_list: (list of sets) The input list to be written.
      filename: (str) Destination file.

    Raises:
      FileError: Unable to read filename.
      ValueError: Movie list is empty.
    """

    if not image_list:
        raise ValueError('Image list is empty')
    logging.info('Saving Image results in %s.', filename)
    # Write results into a file.
    with open(filename, 'w+') as csv_file:
        csv.writer(csv_file).writerows(image_list)


def ExtractImageData(filename):
    """Extract content data from Image.

    Args:
      filename: (str) File of the Image.

    Returns:
      Image content as str.
    """

    with open(filename) as image_file:
        return image_file.read()


def GetImages(image_list):
    """Get a list of file location for each Image in ImageList.

    This method generates a list of Image objects, ArtWorkImage contains
    the Image master_id and path where the Image is located.

    Args:
      image_list: list.

    Returns:
      A collections.Sequence of ArtWorkImage namedtuples.

    Raises:
      ValueError: No Images records found.
    """

    if not image_list:
        raise ValueError('Invalid images')
    return [ArtWorkImage(*image) for image in image_list]


def _ExtractLabels(image):
    """Extract labels from Image using Google Cloud Vision API.

    Args:
      image: (ArtWorkImage) MovieImage object.

    Returns:
      list: List of labels found in Movie image.
    """

    # Initialize Cloud Vision client.
    service = _GetService(FLAGS.api_key)
    if not image.path:
        logging.error('No movie path found for: %s.', image.master_id)
        return
    image_data = ExtractImageData(image.path)
    batch_request = [{
        'image': {
            'content': base64.b64encode(image_data).decode('UTF-8')
        },
        'features': [{
            'type': 'LABEL_DETECTION',
            'maxResults': _MAX_RESULTS,
        }]
    }]
    request = service.images().annotate(body={'requests': batch_request})
    # Send API request to Google Cloud Vision API.
    response = request.execute(num_retries=_NUM_RETRIES)
    return _HandleApiResponse(response)


def _HandleApiResponse(response):
    """Process JSON from API response.

    Args:
      response: (str): string in JSON format.

    Returns:
      A list of labels as a 'bag of words' or empty string if no labels found.

    Raises:
      ValueError: No response.
    """

    if not response:
        raise ValueError('No response')
    if _RESPONSES not in response:
        logging.error('No responses found.')
        return []
    if len(response[_RESPONSES]) != 1:
        logging.error('No labelAnnotations value in responses.')
        return []
    annotations = response[_RESPONSES][0]
    text_labels = []
    # Extract Label annotations.
    if _ANNOTATIONS in annotations:
        for annotation in annotations[_ANNOTATIONS]:
            if _DESCRIPTION in annotation:
                text_labels.append(annotation[_DESCRIPTION])
    logging.info('Number of labels found: %d.', len(text_labels))
    return _GenerateBagOfWords(text_labels)


def _GenerateBagOfWords(list_of_words):
    """Generates a bag of words from a word list.

    Args:
      list_of_words: (list) The list of words (unicode or str).

    Returns:
      (str): List of words as string comma separated.
    """
    if not list_of_words:
        return ''
    # Generate a bag of words from list using comma as default separator.
    return ','.join(
        s.encode('utf-8') if isinstance(s, unicode) else str(s)
        for s in list_of_words)


def ProcessImageList(image_list):
    """Extract labels from image, image resides in CNS or local file.

    Args:
      movies: (list). Contains movie information.

    Returns:
      list: List of tuples (master_id, image labels).

    Raises:
        ValueError: Invalid movie image list.
    """

    if not image_list:
        raise ValueError('Invalid image list')

    image_labels = []
    with futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        # Start the load operations and mark each future with its Image id.
        future_to_url = {
            executor.submit(_ExtractLabels, image): image for image in
            image_list}
        for future in futures.as_completed(future_to_url):
            image = future_to_url[future]
            try:
                image_labels.append((image.master_id, future.result()))
            except ValueError as e:
                logging.exception('Exception at: %s, %s', image.master_id, e)
    return image_labels


def AnalyzeLabels(dataframe):
    """Analyze labels in text.

    Format:
    MASTER_ID, LABELS
    TheArsenal_1928.png,"art,mural,painting,artwork,recreation"

    :param dataframe:
    :return:
    """
    entities = {}
    for _, labels in dataframe.iterrows():
        for label in labels[1].split(','):
            if entities.get(label):
                entities[label] += 1
            else:
                entities[label] = 1
    return entities


def main(_):
    """Reads images list from folder.
      Generates a list with Image path.
      For each image in list get Labels from Cloud Vision API.
      Write results in file.
    :param _:
    :return:
    """
    # Read movie information from File.
    logging.info('Extracting Images path information from: %s.', FLAGS.folder)
    images = LoadDataSet(FLAGS.folder)
    # Extract labels from Art images via Cloud Vision API.
    image_list = GetImages(images)
    image_labels = ProcessImageList(image_list)
    # Save results.
    file_results = FLAGS.results or _FILE_RESULTS
    SaveDataSet(image_labels, file_results)
    if FLAGS.graph:
        logging.info('Analyzing images from %s.', file_results)
        labels = LoadResultsFile(file_results)
        if _IsEmpty(labels):
            raise ValueError('No records found')
        else:
            entities = AnalyzeLabels(labels)
            logging.info(entities)
    logging.info('Analysis completed.')

if __name__ == '__main__':
    app.run(main)
