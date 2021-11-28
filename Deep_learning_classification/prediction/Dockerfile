FROM python:3.7.5-buster
FROM tensorflow/tensorflow

RUN echo \
   && apt-get update \
   && apt-get --yes install apt-file \
   && apt-file update
RUN echo \
   && apt-get --yes install build-essential

RUN pip3 install --upgrade pip
RUN pip install --upgrade pip setuptools wheel

COPY . /app
WORKDIR /app

RUN pip3 --no-cache-dir install -r requirements.txt

RUN pip3 install tensorflow
RUN pip3 install keras


CMD ["python","predict.py"]