# Instruction to reproduce paper results

Below, we describe the steps to reproduce the results from our paper "Spreading of COVID-19 in Brazil: Impacts and uncertainties in social distancing strategies". This guideline is tested only in GNU/Linux OS, more precisely, we tested it in Ubuntu 18.04.4 LTS.

## Creating a virtual environment and installing dependencies

First, we have to assure that we are using the same versions for all packages. This is important to achieve the same results we obtained.
The following steps will create (nearly) the same setup as the one we have tested.

1. Install `python > 3.6`, in case that you didn't install it yet. Please be careful about Python version. We use Python 3.7.

2. Install `virtualenv`:
    ```console
    $ pip3 install virtualenv
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
    $ pip3 install -r requirements-minimal.txt
    ```

## Running case studies

To run either BR or RJ case, please follow the steps below carefully:

1. In the case that you didn't activate your virtual environment, please do it with:
    ```console
    $ source .env/bin/activate
    ```

2. Create a output directory to save the results. For instance, you can set `XX = BR` or `XX = RJ`, set it according to the case you want to run.
    ```console
    $ mkdir output_dir_XX ; cd output_dir_XX
    ```

3. From the directory created in the previous step, run the script with `xx = br` or `xx = rj` according to the case you want to run.
    ```console
    $ python ../scripts/xx_model_bayes_SMC.py
    ```

4. After the simulation from the previous step ends (without problems), you can run post-processing routine to plot the figures. As the previous step, use `xx = br` or `xx = rj` according to the case.
    ```console
    $ python ../scripts/post_processing_xx.py
    ```

Done! You should be able to obtain the same results we did from the previous steps. No errors are expected between each step. If you find any error, please contact us and we will try to help you.

**P.S.:** Be aware the the simulation scripts use parallel processing. The scripts will automaticaly detect the number of available CPU you have in your machine and they will use all the CPUs available.