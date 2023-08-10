FROM python:3.9

WORKDIR /code

# COPY ./requirements.txt /code/requirements.txt

COPY . .
RUN pip install --no-cache-dir --upgrade ".[streamlit]"


ENV DOCKER_CONTAINER=1

CMD ["streamlit", "run", "app.py", "--server.port", "7860"]