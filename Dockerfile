# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY data data

COPY predict.py predict.py

COPY predict_covid predict_covid

COPY api.py api.py

EXPOSE 8000/tcp

CMD ["flask", "run", "--host", "0.0.0.0", "--port", "8000"]

