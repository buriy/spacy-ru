# Модель русского языка для библиотеки spaCy

## Преимущества модели ru2
Модель ru2 умеет определять не только POS-tag в x.pos_, но и лемму слова в x.lemma_ . Например, для существительных, лемма совпадает с формой именительного падежа, единственного числа.
Из-за особенностей устройства библиотеки spacy, для более хорошего качества лемм, нужно писать
```
import ru2
nlp = ru2.load_ru2('ru2')
```
вместо стандартного
```
import spacy
nlp = spacy.load('ru2')
```

Это также починит `.noun_chunks()` для русского, но они пока не идеально работают, будем доделывать.

## Модель ru2e
Это "пустая" модель, которая использует стемминг (`pip install pystemmer`), полезна для пользовательских задач классификации, особенно, когда данных мало. Поскольку в этой модели нет POS-теггера, она не умеет получать леммы.
Для использования стемминга надо писать аналогично модели ru2 выше:
```
import ru2e
nlp = ru2e.load_ru2('ru2e')
```

Смотри пример в 
https://github.com/buriy/spacy-ru/blob/master/notebooks/examples/textcat_news_topics.ipynb

# Установка

Инсталляция сейчас не супер-простая, кроме того, thinc не всегда из коробки работает.
Зависимости из проекта spacy-ru нужны только если вы собираетесь повторять обучение моделей для spacy-ru или повторять ноутбуки.

## Установка и использование ru2 модели:
*для примера установки модели в окружении conda, вы можете ознакомится с [Dockerfile](Dockerfile)*
1. установить pymorphy2==0.8
- pip: `pip install pymorphy2==0.8`
- conda: *к сожалению в репозиторях anaconda данный пакет доступен только для платформы osx-64* `conda install -c romanp pymorphy2==0.8`	

2. установить spacy 2.1:
- pip: `pip install spacy==2.1.9`
- conda: `conda install -c conda-forge spacy==2.1.9`
3. Скопировать каталог ru2 из репозитория себе в проект: `git clone -b v2.1 https://github.com/buriy/spacy-ru.git && cp -r ./spacy-ru/ru2/. /my_project_destination/ru2 `
 
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
 - Если нужна работа на GPU (ускоряет обучение в 2-3 раза, инференс -- до 5 раз), то, возможно, нужно исправить (явно указать) путь к cuda и переустановить библиотеку thinc:
```bash
pip uninstall -y thinc
CUDA_HOME=/usr/local/cuda pip install --no-cache-dir thinc==7.0.8
```
Другой вариант -- попробовать что-то типа `pip install "spacy[cuda91]<2.2"` или `pip install "spacy[cuda10]<2.2"` для spacy версии 2.1.x и вашей версии cuda.

Если GPU по-прежнему не работает -- стоит явно проверить, что `cupy` установлена верно для вашей версии cuda: [link](https://docs-cupy.chainer.org/en/stable/install.html#install-cupy)
пример установки для cuda 10.0
```bash
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
$ which nvcc
/usr/local/cuda/bin/nvcc
$ CUDA_HOME=/usr/local/cuda
$ pip install cupy-cuda100
...
Successfully installed cupy-cuda100-7.1.0
$ pip install --no-cache-dir "spacy[cuda10]<2.2"
...
Successfully installed blis-0.2.4 preshed-2.0.1 spacy-2.1.9 thinc-7.0.8
```

- Если вы переходите с многоязычной модели "xx" на модели ru/ru2, то имейте в виду, что токенизация в моделях ru/ru2 и xx отличается, т.к. xx не отделяет буквы от цифр и дефисы, то есть скажем слова "24-часовой" и "va-sya103" будут едиными неделимыми токенами.
- На Windows клонирование репозитория с настройкой `core.autocrlf true` в `git` 
может испортить некоторые файлы и привести к ошибкам типа `msgpack._cmsgpack.unpackbTypeError: unhashable type: 'list'`.
Для того чтобы этого избежать надо либо клонировать с `core.autocrlf false`, либо, например, 
скачивать архив репозитория вручную через веб-интерфейс.
Обсуждение проблемы и решение можно найти [здесь](https://github.com/explosion/spaCy/issues/1634).
- Попытка вызова `spacy.displacy.serve()` или некоторых других функций на Python 3 может привести к 
ошибке `TypeError: __init__() got an unexpected keyword argument 'encoding'`. Чтобы этого избежать, раньше
необходимо было явно установить старую версию `msgpack-numpy<0.4.4.0`. Сейчас вроде бы поправили. Обсуждение проблемы и решение можно найти [здесь](https://github.com/explosion/spaCy/issues/2810).
