# Shortest Path RL

Training a DRL agent for solving the shortest path problem in bi-directional cyclic graphs.

## Installation

[Poetry](https://python-poetry.org/) is used for dependency management. So please install poetry:

```bash
$ curl -sSL https://install.python-poetry.org | python3 -

```

To install all the dependencies, please enter the following from the project's root directory:

```bash
$ poetry install

```

Then enter the virtual environment:

```bash
$ poetry shell

```

For random policy:

```bash
$ python random_policy.py

```

For training the DQL model:

```bash
$ python dql_train.py --verbose --plot

```
