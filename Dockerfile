FROM continuumio/miniconda3
LABEL maintainer="https://github.com/ex00/"

ENV PROJECT_DIR=/example_spacy_ru_project
RUN mkdir $PROJECT_DIR

WORKDIR /
#istall components for ru2 
RUN conda install -y -c conda-forge spacy==2.0.12
RUN pip install pymorphy2==0.8
RUN git clone https://github.com/buriy/spacy-ru.git
RUN cp -r /spacy-ru/ru2/. $PROJECT_DIR/ru2


ADD  ./examples/full_simple_example.py $PROJECT_DIR/
RUN conda install -y -c conda-forge pandas tabulate # install packages for example 
WORKDIR $PROJECT_DIR
CMD python full_simple_example.py
