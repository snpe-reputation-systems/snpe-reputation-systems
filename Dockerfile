FROM python:3.9.16

# Install poetry (If executed by hand works fine, but not in dockerfile)
RUN apt-get update \ 
    && apt-get install -y curl \
    && curl -sSL https://install.python-poetry.org | python3 - 

RUN mkdir /snpe-package
WORKDIR /snpe-package 
COPY pyproject.toml /snpe-package/

RUN export PATH="/root/.local/bin:$PATH" \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

RUN pip install Jupyter

# Make a data folder which will be connected to the host






