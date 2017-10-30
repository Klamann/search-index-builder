#!/usr/bin/env python
from setuptools import setup

# required libs (on debian):
#   apt-get install libxml2-dev libxslt1-dev
#
# get the spacy model after installation:
#   python3 -m spacy download en

setup(
    name='search-index-builder',
    version='1.0.0',
    description='a set of scripts to build a topic model and a search index from a document collection',
    author='Sebastian Straub',
    author_email='sstraub (at) posteo (dot) de',
    url='https://github.com/Klamann/search-index-builder',
    packages=[],
    license='Apache 2.0',
    classifiers=(
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ),
    install_requires=[
        'sickle>=0.5',
        'pdfminer.six>=20170419',
        'elasticsearch>=5.3.0,<6.0.0',
        'xmltodict>=0.10.0',
        'langdetect>=1.0.7',
        'numpy>=1.7',
        'spacy>=1.8.0,<2.0',
        'gensim>=2.1.0,<3.0',
        'requests>=2.12,<3.0',
        'pdfminer.six>=20170419',
    ],
)
