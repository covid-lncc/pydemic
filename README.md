# Pydemic: a package for disease spreading models

[![License: LGPL3](https://img.shields.io/badge/license-LGPL%20(%3E%3D%203)-blue)](https://opensource.org/licenses/LGPL-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linux](https://github.com/covid-lncc/pydemic/workflows/linux/badge.svg?branch=master)](https://github.com/covid-lncc/pydemic/actions?query=workflow%3Alinux)
[![windows](https://github.com/covid-lncc/pydemic/workflows/windows/badge.svg?branch=master)](https://github.com/covid-lncc/pydemic/actions?query=workflow%3Awindows)
<!--- [![badge](https://img.shields.io/badge/try%20it%20in%20your%20browser-binder-579ACA.svg?logo=data:image/png;base64,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)](https://mybinder.org/v2/gh/covid-lncc/pydemic/master) --->

## What is it?

A package to model disease spreading with SIR/SEIR-based models focused on SARS-CoV-2 (COVID-19).
We are currently in a very early development stage.

## Development mode

In the following, we describe how you can use `pydemic`.

### First time

Well, if this is your first time, first you need to download `pydemic`. It is currently not distributed
by PyPA, so you have to clone this repository:

    $ git clone https://github.com/covid-lncc/pydemic.git

Then navigate to the package directory:

    $ cd pydemic

Now, in order to develop and use `pydemic`, you have to execute the following steps:

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
    $ inv hooks
    ```

Now you can properly use `pydemic`. The steps 1-3 and 6 are necessary only in the first time
that you configured to develop `pydemic`. Step 5 is needed when new dependencies
are added. If you had executed Step 5 before, you can simply do `inv requirements` in your terminal
every time you add new dependencies to `requirements.txt` file.

### Development daily tips

Below, some tips that help me while developing `pydemic`:

* You modified the package and added new tests, how could you run the tests? Simply run the
following:

        $ inv tests

* Do you want to install `pydemic` in active Python site-packages? No time to loose, just do:

        $ inv devinstall

* Ah, so you just want a full and up-to-date `.csv` file with world-wide record for COVID-19?
You can have it:

        $ inv generatedb

There are more tasks available at `tasks.py`. Please feel free to have a look at it.

## Basic usage

Please check our [notebooks](https://github.com/covid-lncc/pydemic/tree/master/notebooks),
basic demos are provided there.

## Contact

My name is Diego. Feel free to contact me through the email <volpatto@lncc.br>.
