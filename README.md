Shortest Path RL
Training a DRL agent for solving the shortest path problem in bi-directional cyclic graphs.

Installation
Poetry is used for dependency management. So please install poetry:

$ curl -sSL https://install.python-poetry.org | python3 -
To install all the dependencies, please enter the following from the project's root directory:

$ poetry install
Then enter the virtual environment:

$ poetry shell
For random policy:

$ python random_policy.py
For training the DQL model:

$ python dql_train.py --verbose --plot