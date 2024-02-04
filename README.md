# SNPE Reputation Systems

## Introduction

The goal of this repository is to adapt the work previously developed on its [parent repository](https://github.com/narendramukherjee/reputation-systems/tree/master) to produce an installable Python package that provides a more convenient way to interact with the SNPE-reputation-systems project. Achieving this goal involved three main tasks:

1. Design and implement unit tests for the parent repository's original modules that ensured code integrity and consistency before packaging.
2. Provide the necessary components to allow the repository to be packaged through the use of Python's packaging framework [Poetry](https://python-poetry.org/).
3. Set up a [GitHub Actions-based](https://tilburgsciencehub.com/topics/automation/automation-tools/deployment/intro_ghactions/) CI pipeline that can perform the following tasks in an automated manner whenever the package is updated:
    - Run unit tests and format checks.
    - Conduct packaging process.


## 1. Code testing

The unit tests of the repository are implemented through the `Pytest` and `Hypothesis` Python testing frameworks. More particularly, while `Pytest` takes care of the execution of the tests and their structure, the individual tests are built using `Hypothesis`.

Among the different modules of the [parent repository](https://github.com/narendramukherjee/reputation-systems/tree/master), at the moment of writing this Readme, unit tests have been implemented for `simulator_class.py`. These can be found inside the `snpe_reputation_systems/tests/` subdirectory, where `test_simulator_class.py` is the main testing file. Additionally, other Python scripts containing alternative unit test implementations are found within `snpe_reputation_systems/tests/`. These are not part of the testing workflow but were preserved in case they could be useful in future rounds of unit test implementation. These files are `legacy_test_simulator_class.py`, which contains discarded tests from old versions of `test_simulator_class.py`, and `inheritance_test_simulator_class.py`, including a discarded proposal to structure tests following subsequent parent-child classes. The latter was finally not implemented due to compatibility issues between the class structure for tests and the `Hypothesis` framework.

Additionally, beyond unit tests for ensuring code integrity, format checks are also run over the code modules to ensure consistency in coding standards and readability across the project. These format checks are run as part of the main testing and packaging workflow described below.

## 2. Packaging with Poetry

The `pyproject.toml` file is the primary configuration file for managing project dependencies, package metadata, and the build system via Poetry. Here’s a breakdown of its key sections:

- **[tool.poetry]**: Specifies the package name, version, description, authors, and other metadata. This information is used for identifying the package and its purpose when published.

- **[tool.poetry.dependencies]**: Lists the required packages and versions needed for our package to function properly. We use the caret (^) symbol to allow updates that do not break semantic versioning and asterisks (*) for dependencies to get the latest version. Specific versions of numpy are pinned to prevent issues with newer versions.

- **[tool.poetry.group.test.dependencies]**: Defines dependencies needed for testing but not for the actual package use, like pytest and hypothesis. These are separated into their own group so they can be installed only when needed.

- **[build-system]**: Designates poetry-core as the build backend, making it responsible for packaging and distribution.

The configuration ensures that the package is built with the correct dependencies and metadata, following the Python packaging standards.

## 3. GitHub Actions Workflow

The repository's main workflow, defined in `.github/workflows/formatting-testing-packaging.yml`, is divided into three segments, or jobs, as they are referred to in the context of GitHub Actions. Before reviewing these one by one, if you are new to GitHub Action we advise you to check the following [article introducing the GitHub Actions platform](https://tilburgsciencehub.com/topics/automation/automation-tools/deployment/intro_ghactions/).

### 1. `format-check`
Defined in lines 10 through 28 of `formatting-testing-packaging.yml`, this job scans the codebase to ensure that all Python code complies with the code formatting standards of `black`, `mypy`, and `isort` by making use of the `super-linter` action. This action generates a code formatting report that can be consulted as part of the workflow execution details. It will also emit a warning message in case any formatting issues are detected. If there are formatting issues in the code, the execution of the workflow is interrupted; otherwise, the execution proceeds with the next job.

The job involves the following steps:
- **Checkout Repository**: Checks out the repository to make it available for use within the runner.
- **Super-linter**: Runs the `super-linter` action to scan for formatting inconsistencies and generate the code formatting report.

### 2. `code-testing`
Defined in lines 30 through 43 of `formatting-testing-packaging.yml`, It is executed upon successful completion of `format-check` to run the codebase's unit tests. This second job takes place inside a self-hosted runner, which must be active for the job to be executed. For an introduction to self-hosted runners, we recommend you visit [the following article on the topic](https://tilburgsciencehub.com/topics/automation/automation-tools/deployment/ghactions-self-hosted-runner/). Additionally, below you will find more details about the implementations of self-hosted runners in this project. In a similar manner to `format-check`, if any of the unit tests fail, the execution of the workflow will be aborted and a warning message will be emitted so the appropriate fixes can be implemented.

The job involves the following steps:
- **Checkout Repository**: Checks out the repository to make it available for use within the runner.
- **Run Tests**: Executes the command `pytest` which invokes the execution of all unit tests within the project following the pytest framework.

### 3. `build-and-publish-package`
Defined in lines 45 through 68 of `formatting-testing-packaging.yml`, if the previous jobs were completed successfully, `build-and-publish-package` will finally take care of generating a new version of the project's package through Poetry and publish it as an artifact within the workflow.

The job involves the following steps:
- **Checkout Repository**: Checks out the repository to make it available for use within the runner.
- **Set Up Python**: Sets up the specified Python version.
- **Install Poetry**: Installs Poetry, our packaging tool.
- **Build Package**: Executes the `poetry build` command to create the package.
- **Publish Package as Artifact**: The built package is published as an artifact within the workflow.


## Installing the package

Once built, the package can be installed using one of the following methods:

### Option 1: Installing from GitHub Artifacts

1. Go to the Actions tab in our GitHub repository.
2. Select the workflow run you want.
3. Download the artifact named `snpe_reputation_systems_package` from the run.
4. Install the downloaded `.whl` file using pip with the command: 

```bash
pip install path_to_downloaded_wheel.whl
```

This method is particularly useful when you want to install a specific version of the package that may not be the latest commit on the main branch. It provides fine-grained control over which build to install.

### Option 2: Installing via Git

You can also install the package directly using Git, which will fetch the latest version from the main branch of the repository. The command for this installation is:

```bash
pip install git+https://github.com/snpe-reputation-systems/snpe-reputation-systems.git
```

This method ensures that you are always working with the most recent version of the code that has been pushed to the main branch. It's a good option for those who want to stay up-to-date with the latest developments of the project.

Note: if you are working on a branch, and want to retrieve the latest version pushed to that branch, you should add '@branch-name' at the end of the install command, as seen in the picture below:

![Succesful package installation from the branch](images/install-package-opt2.png)

## Self-hosted runners

As mentioned above, the testing job of the main workflow is executed in a self-hosted runner. The reason for this is that some unit tests can be computationally demanding and GitHub-provided runners may not be able to handle them. 

All the relevant information to understand self-hosted runners and work with them within the context of this project can be found [here]. Particularly, this project follows the "Docker-based approach" described in the article in combination with the Dockerfile template which can be found inside the `self_hosted_runner/` sub-directory of the repository.

## Familiarize yourself with the project

To familiarize yourself with the content of the project, we invite you to check the [tutorial notebooks available in the parent repository](https://github.com/narendramukherjee/reputation-systems/tree/master/snpe/tutorials). These cover the three main modules of the project: non-marketplace simulations, marketplace simulations, and inference.