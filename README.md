# Модель русского языка для библиотеки spaCy

## Преимущества модели ru2
Оно старается определять не только x.pos_, но и x.lemma_ -- лемму слова (например, для существительных лемма совпадает с формой: "именительный падеж, единственное число") (edited) 
# Установка

Инсталляция сейчас не супер-простая, кроме того, thinc не всегда из коробки работает.
Зависимости из проекта spacy-ru нужны только если вы собираетесь повторять обучение моделей для spacy-ru или повторять ноутбуки.

## Установка и использование ru2 модели:
*для примера установки модели в окружении conda, вы можете ознакомится с [Dockerfile](Dockerfile)*
1. установить pymorphy2==0.8
- pip: `pip install pymorphy2==0.8`
- conda: *к сожалению в репозиторях anaconda данный пакет доступен только для платформы osx-64* `conda install -c romanp pymorphy2==0.8`	

2. установить spacy==2.0.12
- pip: `pip install spacy==2.0.12`
- conda: `conda install --c conda-forge spacy==2.0.12`
3. Скопировать каталог ru2 из репозитория себе в проект: `git clone https://github.com/buriy/spacy-ru.git && cp -r ./spacy-ru/ru2/. /my_project_distination/ru2 `
 
После этого нужно загрузить модели с морфологией и синтаксисом 
```python
import spacy
sample_sentences = "Привет Миру! Как твои дела? Сегодня неплохая погода."
if __name__ == '__main__':
    nlp = spacy.load('ru2')
    nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
    doc = nlp(sample_sentences)
    for s in doc.sents:
    	print(list(['lemma "{}" from text "{}"'.format(t.lemma_, t.text) for t in s]))
``` 
Если нужна модель с pymorphy2 в качестве лемматизатора и POS: `nlp = spacy.load('ru2', disable=['tagger', 'parser', 'NER'])`

### Пример в Docker контейнере
вы можете попробовать пример использования ru2 модели:
```bash
git clone https://github.com/buriy/spacy-ru.git
cd spacy-ru
docker build -t spacy:ru2 .
docker run --rm spacy:ru2
```

### Предупреждения и возможные проблемы
 - Если нужен работающий thinc на GPU, то, возможно, нужно исправить (явно указать) путь к cuda и переустановить библиотеку:
```bash
pip uninstall -y thinc
CUDA_HOME=/usr/local/cuda pip install --no-cache-dir thinc\<6.11
```
- Если вы переходите с xx на ru/ru2, то имейте в виду, что токенизация в ru/ru2 и xx отличается, т.к. xx не отделяет буквы от цифр и дефисы.
- На Windows клонирование репозитория с настройкой `core.autocrlf true` в `git` 
может испортить некоторые файлы и привести к ошибкам типа `msgpack._cmsgpack.unpackbTypeError: unhashable type: 'list'`.
Для того чтобы этого избежать надо либо клонировать с `core.autocrlf false`, либо, например, 
скачивать архив репозитория вручную через веб-интерфейс.
Обсуждение проблемы и решение можно найти [здесь](https://github.com/explosion/spaCy/issues/1634).
- Попытка вызова `spacy.displacy.serve()` или некоторых других функций на Python 3 может привести к 
ошибке `TypeError: __init__() got an unexpected keyword argument 'encoding'`. Чтобы этого избежать,
необходимо явно установить старую версию `msgpack-numpy<0.4.4.0`. Обсуждение проблемы и решение можно
найти [здесь](https://github.com/explosion/spaCy/issues/2810).
