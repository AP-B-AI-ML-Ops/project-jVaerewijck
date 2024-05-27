FROM python:3.11-slim

RUN pip install pip -U
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile" , "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py" , "./" ]
COPY [ "./models/lin_reg.bin" , "./models/" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn" , "--bind=0.0.0.0:9696" , "predict:app" ]