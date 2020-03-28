# Pydemic: a package for disease spreading models

[![License: LGPL3](https://img.shields.io/badge/license-LGPL%20(%3E%3D%203)-blue)](https://opensource.org/licenses/LGPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![](https://github.com/covid-lncc/pydemic/workflows/linux/badge.svg?branch=master)
![](https://github.com/covid-lncc/pydemic/workflows/osx/badge.svg?branch=master)
![](https://github.com/covid-lncc/pydemic/workflows/windows/badge.svg?branch=master)

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
