from setuptools import setup

setup(
   name='spacy-ru',
   version='0.2.1',
   packages=['ru2'],
   author='Yuri Baburov',
   author_email='burchik@gmail.com',
   install_requires=['spacy>=2.0', 'pymorphy2>=0.8']
)