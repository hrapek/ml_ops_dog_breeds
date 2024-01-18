import os

#_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
#_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
#_PATH_DATA = os.path.join(_PROJECT_ROOT, 'data')  # root of data

#_DATASCRIPTS_ROOT = os.path.dirname(__file__)  # folder of this script == ml_ops_dog_breeds/ml_ops_dog_breeds/data
#_PROJECT_ROOT = os.path.dirname(_DATASCRIPTS_ROOT)  # root of project == ml_ops_dog_breeds/ml_ops_dog_breeds
#_FOLDER_ROOT = os.path.dirname(_PROJECT_ROOT)  # root of folder == ml_ops_dog_breeds
#_PATH_DATA = os.path.join(_FOLDER_ROOT, 'data')  # path of data (not this data folder, but global)

_PATH_ROOTFOLDER = os.path.join(os.path.dirname(__file__).split('ml_ops_dog_breeds')[0], 'ml_ops_dog_breeds')
_PATH_DATA = os.path.join(_PATH_ROOTFOLDER, 'data')
