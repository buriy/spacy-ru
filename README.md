### To install:

```
sudo pip install poetry
git clone https://github.com/buriy/spacy-ru.git
cd spacy-ru
virtualenv --python=python3.6 .venv
poetry install
```
### To install for Windows:

```
git clone https://github.com/buriy/spacy-ru.git
cd spacy-ru
virtualenv --python=python3.6 .venv
python setup.py install 
```

### Run example:

To activate a virtualenv in a shell:
```
. .venv/bin/activate
```

Then, to run an example russian pipeline:

```
./tokenize-ru.sh
```
To run an example multilanguage pipeline (for a comparison):

```
spacy download xx_ent_wiki_sm
./tokenize-xx.sh
```
