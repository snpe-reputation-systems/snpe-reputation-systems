FROM continuumio/miniconda3

RUN mkdir actions-runner; cd actions-runner

RUN apt-get update && apt-get install -y curl

WORKDIR /actions-runner
RUN curl -O -L https://github.com/actions/runner/releases/download/v2.304.0/actions-runner-linux-x64-2.304.0.tar.gz

RUN tar xzf ./actions-runner-linux-x64-2.304.0.tar.gz

RUN ./bin/installdependencies.sh

RUN useradd -m runneruser
RUN chown -R runneruser:runneruser /actions-runner
USER runneruser

RUN  ./config.sh --url <REPO_URL> --token <TOKEN> --work --unattended --replace --name <RUNNER_NAME>

USER root
RUN conda install pytorch torchvision torchaudio -c pytorch
RUN pip install sbi hyperopt mlflow tqdm arviz pytest hypothesis


COPY entrypoint.sh /actions-runner
RUN chown runneruser:runneruser /actions-runner/entrypoint.sh

USER runneruser
RUN chmod +x /actions-runner/entrypoint.sh
ENTRYPOINT ["/actions-runner/entrypoint.sh"]







