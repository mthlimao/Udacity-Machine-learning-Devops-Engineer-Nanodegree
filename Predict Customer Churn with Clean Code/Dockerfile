FROM python:3.8-bookworm

ENV DEVENV_PATH=/udacity
ENV CONTAINER_PROJECT_PATH=${DEVENV_PATH}/clean_code
ENV PYTHONPATH "${PYTHONPATH}:${CONTAINER_PROJECT_PATH}"

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        # deps for building python deps
        build-essential \
        unzip \
        libaio1 \
        alien \
        iputils-ping && \
        apt clean

WORKDIR ${CONTAINER_PROJECT_PATH}

COPY ./requirements.txt ./requirements.txt

RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt 

COPY ./data/ ./data/
COPY ./source/ ./source/
COPY ./notebooks/ ./notebooks/
RUN mkdir -p ./images/ && mkdir -p ./models/ && mkdir -p ./logs/

CMD ["python", "source/churn_script_logging_and_tests.py"]
