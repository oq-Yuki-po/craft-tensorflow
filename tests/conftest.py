import pathlib
import shutil

import h5py
import pytest

OUTPUT_PATH = 'tests/test_output'


@pytest.fixture(scope='session', autouse=True)
def make_tmp_dir(request):

    pathlib.Path(OUTPUT_PATH).mkdir(exist_ok=True)

    def remove_tmp_dir():
        shutil.rmtree(OUTPUT_PATH)

    request.addfinalizer(remove_tmp_dir)
    return make_tmp_dir


@pytest.fixture
def load_synthtext():
    """synthtextの読み込み
    """
    with h5py.File('tests/dataset/SynthText.h5', 'r') as db:

        db_data = sorted(db['data'].keys())
        image_name = db_data[0]

        image = db['data'][image_name][...]
        charBB = db['data'][image_name].attrs['charBB']
        texts = db['data'][image_name].attrs['txt']

        texts = [i.decode('utf-8') for i in texts]
        character_bbox = charBB.transpose(2, 1, 0)

        characters_list = []

        for i in texts:
            for j in i.split('\n'):
                characters = []
                for k in list(j):
                    if k != ' ':
                        characters.append(k)

                characters_list.append(characters)

    return image, character_bbox, characters_list
