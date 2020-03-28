# Pydemic: a package for disease spreading models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Build Status](https://travis-ci.com/volpatto/blank-python-project.svg?branch=master)](https://travis-ci.com/volpatto/blank-python-project)
![](https://github.com/volpatto/blank-python-project/workflows/linux/badge.svg?branch=master)
![](https://github.com/volpatto/blank-python-project/workflows/osx/badge.svg?branch=master)
![](https://github.com/volpatto/blank-python-project/workflows/windows/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/volpatto/blank-python-project/branch/master/graph/badge.svg)](https://codecov.io/gh/volpatto/blank-python-project)
[![Documentation Status](https://readthedocs.org/projects/blank-python-project/badge/?version=latest)](https://blank-python-project.readthedocs.io/en/latest/?badge=latest)

## What is it?

A package to model disease spreading with SIR/SEIR-based models. We are currently in a very early development stage.

## Development mode

In order to develop and use `pydemic`, you have to execute the following steps:

1. Install `python > 3.6`.

2. Install `virtualenv`:
    ```console
    $ pip install virtualenv
    ```

3. Create your isolated environment (here named as `.env`):
    ```console
    $ virtualenv .env
    ```

4. Activate the environment:
    ```console
    $ source .env/bin/activate
    ```

5. Install all required dependencies:
    ```console
    $ pip install -r requirements.txt
    ```

6. Setup the checkers and formatters:
    ```console
    $ pre-commit install
    ```

Now you can properly use `pydemic`. The steps 1-3 and 6 are necessary only in the first time
that you configured to develop `pydemic`. Step 5 is needed when new dependencies
are added.

## Contact

My name is Diego. Feel free to contact me through the email <volpatto@lncc.br>.
