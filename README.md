## 3D Reconstruction Network
This project will be used to maintain all code for the 3D reconstruction project for the course in ECE 740. There are different stages in this project:
1. **Dataset Design**: Here we will decide the dataset we would like to work on from the tracker that has already been shared. The idea is to find the most convenient dataset to work with and decide the final splits: train, development and test.
    - Train: This split will be used to actually train the network with a valid loss function.
    - Development: Used to evaluate the network and hyperparameter tuning (if time permits)
    - Test: Unseen dataset, ideally from a different source
2. **Training Experiments**: Here we would like to conduct the actual experiments using the splits we have decided on. Each experiment should be clearly documented: the hypothesis being tested, actual results and observation
3. **Documentation**: Final compilation of all our results and create the presentation

## Installation
This repo used `Poetry` as the package manager. Please follow the steps to reproduce the exact environment. This is very important for collaboration since we should all be using the same environment.

### Pyenv Installation

1. Install `pyenv` in the terminal. For MacOS, enter the following instructions:
    `brew update`
    `brew install pyenv`
2. Add to bash profile:
    ```
    $ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
    $ echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile 
    ```
3. Add `pyenv init` to shell:
    `$ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bash_profile`

4. Restart your shell and it should work.

5. I found these instructions helpful for installing `pyenv` [article](https://medium.com/thoughful-shower/how-to-install-python-pyenv-on-macos-e033c4afbba4)

6. To test successful installation enter `pyenv` in the terminal, if you get an error it was not successful.
7. After installation install python version `3.11.5`, by entering the command,
    `pyenv install 3.11.5`
8. Set global environment by `pyenv global 3.11.5`

### Poetry installation
1. Navigate to the terminal and check the Python version using `pyenv`
2. Install `poetry` using the following command:
    `curl -sSL https://install.python-poetry.org | python3 -`
3. Once done verify installation using,
    `poetry --version`
4. Next navigate to the project directory and install the environment using the `.lock` file,
    `cd reconstruction-deep-network`
    `poetry install`
5. By default `poetry` uses the `.lock` file for installation if it exists. Once the environment is created, activate it by,
    `poetry shell`
6. Run the scripts `hello_world.py` to test complete installation. It should not give any errors.

## Development 
1. Please do not commit to the `master` or `development` branch of the repo.
2. When pushing to the repo, branch out of the `development`, create a `feature-branch` and submit a pull request to `development`.
3. `master` will only be updated by merging from `development` and not in any other way.
