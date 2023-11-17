FROM continuumio/miniconda3

ARG POETRY_VERSION=1.0.0 
ENV POETRY_VERSION=$POETRY_VERSION

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

RUN  ./config.sh --url <REPO_URL> --token <TOKEN> --work --unattended --replace --name <RUNNER_NAME> --labels <LABELS_HERE>

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






