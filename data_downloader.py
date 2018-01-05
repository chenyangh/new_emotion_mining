import urllib.request
import tarfile
import os
import zipfile
data_url = 'https://drive.google.com/uc?export=download&id=1pitbhWwo9yPwIQscbTveXSr1N7gIZemB'
urllib.request.urlretrieve(data_url, 'emotion_with_folds.tar.gz')
tar = tarfile.open('emotion_with_folds.tar.gz')
tar.extractall()
os.remove('emotion_with_folds.tar.gz')

glove_url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
glove_directory = 'feature'
if not os.path.exists(glove_directory):
    os.makedirs(glove_directory)
zip_path = os.path.join(glove_directory, 'glove.twitter.27B.zip')
urllib.request.urlretrieve(glove_url, zip_path)

zip_f = zipfile.ZipFile(zip_path)
zip_f.extract(member='glove.twitter.27B.200d.txt', path=glove_directory)
os.remove(zip_path)