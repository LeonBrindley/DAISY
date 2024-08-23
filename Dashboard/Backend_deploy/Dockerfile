FROM python:3.9.3-buster

RUN mkdir -p /app /inference_service
RUN mkdir /tmp/work
WORKDIR /app

RUN apt-get update && apt-get install libpq-dev python3-dev -y

COPY ./poetry.lock ./pyproject.toml ./

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

# Copy app code
COPY ./app ./app
# Copy inference_service code
COPY ./inference_service ./inference_service

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

