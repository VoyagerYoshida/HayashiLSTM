FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-runtime
LABEL maintainer="voyagerwy130 <voyager.yoshida@gmail.com>"

ENV ROOTHOME /root

ENV WORKSPACE /var/www
WORKDIR $WORKSPACE

RUN curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

ENV PATH $ROOTHOME/.poetry/bin:$PATH

COPY pyproject.toml $WORKSPACE
COPY poetry.lock $WORKSPACE

RUN apt-get update && apt-get install -y locales
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP.UTF-8  
ENV LC_ALL ja_JP.UTF-8

RUN pip install certifi --ignore-installed
RUN pip install pyyaml --ignore-installed
RUN poetry config virtualenvs.create false && \
    pip install --upgrade pip && \
    pip install -U setuptools && \
    poetry install -n
    
CMD ["python"]
