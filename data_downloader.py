import urllib.request
import tarfile
import os
data_url = 'https://drive.google.com/uc?export=download&id=1pitbhWwo9yPwIQscbTveXSr1N7gIZemB'
urllib.request.urlretrieve(data_url, 'emotion_with_folds.tar.gz')
tar = tarfile.open('emotion_with_folds.tar.gz')
tar.extractall()
os.remove('emotion_with_folds.tar.gz')