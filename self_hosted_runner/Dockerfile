FROM continuumio/miniconda3

ARG POETRY_VERSION=1.2.0 
ENV POETRY_VERSION=$POETRY_VERSION

RUN conda install -y python=3.9.16

RUN mkdir actions-runner; cd actions-runner && \
    apt-get update && apt-get install -y curl

RUN pip install poetry==$POETRY_VERSION


WORKDIR /actions-runner

RUN curl -O -L https://github.com/actions/runner/releases/download/v2.304.0/actions-runner-linux-x64-2.304.0.tar.gz && \
    tar xzf ./actions-runner-linux-x64-2.304.0.tar.gz && \
    ./bin/installdependencies.sh && \
    useradd -m runneruser && \
    chown -R runneruser:runneruser /actions-runner

USER runneruser

ARG RUNNER_TOKEN
RUN  ./config.sh --url https://github.com/snpe-reputation-systems/snpe-reputation-systems --token $RUNNER_TOKEN --work --unattended --replace --name self-hosted --labels format-test-pack-runner
USER root

RUN conda install pytorch torchvision torchaudio -c pytorch && \
    pip install sbi hyperopt mlflow tqdm arviz pytest hypothesis black isort pytest-mock

RUN poetry config virtualenvs.create false

COPY . /actions-runner/

RUN poetry install --no-interaction --no-ansi
# RUN poetry install $(test "$YOUR_ENV" == "production" && echo "--no-dev") --no-interaction --no-ansi

USER runneruser

RUN chmod +x /actions-runner/run.sh
ENTRYPOINT ["/actions-runner/run.sh"]






