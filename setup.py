from setuptools import setup

# load dependencies from requirements.txt
# based on https://stackoverflow.com/questions/26900328/install-dependencies-from-setup-py
import os

reqpath = f'{os.path.dirname( os.path.realpath( __file__ ) )}/requirements.txt'
reqlist = []

if os.path.isfile( reqpath ):
    with open( reqpath ) as f:
        reqlist = f.read().splitlines()

setup(
    name='HypeR',
    packages=['hyper'],
    description='A simple library of Hyperdimensional Computing',
    version='1.0.0',
    url='https://github.com/cs0lar/HypeR',
    author='cristiano.solarino@gmail.com',
    author_email='cristiano.solarino@gmail.com',
    install_requires=reqlist,
    keywords=[ 'Hyperdimensional Computing' 'AI' ]
)
