FROM python:3.10

EXPOSE 5000

RUN mkdir /app
WORKDIR /app

RUN pip install --upgrade setuptools
RUN pip install numpy
RUN pip install matplotlib
RUN pip install tensorflow==2.15.0

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY .. /app

CMD python app_api.py