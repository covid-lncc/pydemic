## Development mode

In the following, we describe how you can use `pydemic` in development mode.

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