import io
import os
from setuptools import setup, find_packages


def read(fname):
    with io.open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8') as f:
        return f.read()


setup(
    name='cryspy',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    #install_requires=['pyxtal'],
    version='0.9.0',
)