from invoke import task


@task
def clean(c, docs=False, bytecode=False, extra=""):
    patterns = ["build"]
    if docs:
        patterns.append("docs/_build")
    if bytecode:
        patterns.append("**/*.pyc")
    if extra:
        patterns.append(extra)
    for pattern in patterns:
        c.run("rm -rf {}".format(pattern))


@task
def build(c, docs=False):
    c.run("python setup.py build")
    if docs:
        c.run("sphinx-build docs docs/_build")


@task
def install(c):
    c.run("python setup.py install")


@task
def sdist(c):
    c.run("python setup.py sdist")


@task
def wheel(c):
    c.run("python setup.py bdist_wheel")


@task
def devinstall(c):
    c.run("pip install -e .")


@task
def generatedb(c):
    c.run(
        "python -c 'import pydemic.data_collector as dc; dc.export_updated_full_dataset_from_jhu()'"
    )


@task
def tests(c):
    c.run("pytest . -n auto -vv")
