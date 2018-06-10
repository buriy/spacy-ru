To install:

sudo pip install poetry

```
git clone https://github.com/buriy/spacy-ru.git
cd spacy-ru
poetry install
```
To run an example russian pipeline:

```
./tokenize-ru.sh
```
To run an example multilanguage pipeline (for comparison):

```
spacy download xx_ent_wiki_sm
./tokenize-xx.sh
```
