[![Click to access video lectures](https://i.imgur.com/OpDza8d.png)](https://wandb.me/m4ml-videos)

# Math for Machine Learning

This short course introduces the core concepts and intuitions
of the three most important branches of mathematics
for machine learning:
linear algebra,
calculus,
and probability.

These notebooks are supplementary to
a [YouTube lecture series](https://www.youtube.com/watch?v=uZeDTwWcnuY&list=PLD80i8An1OEGZ2tYimemzwC3xqkU0jKUg)
and presume proficiency with Python.

The notebooks can be executed in any
of the following ways.

## Colab - Persistent One-Click Version

| Notebook    | Link |
|-------------|------|
| Linear Algebra  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/m4ml-linalg-colab) |
| Calculus  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/m4ml-calc-colab) |
| Probability | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://wandb.me/m4ml-prob-colab) |

Click the badges in the table above to access the exercises as
[Google Colab notebooks](https://research.google.com/colaboratory/).
The only requirement is a Google account.
If you save your work to your own Google Drive,
you can return where you left off.

## Binder - Temporary One-Click Version

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/wandb/edu/main?filepath=math-for-ml)

If you click the badge above,
you'll launch a free cloud server
provided by the
[Binder project](https://mybinder.readthedocs.io/en/latest/)
with the appropriate computational environment.
There is no need to create an account.
This environment is _ephemeral_,
or temporary:
after 10 minutes of inactivity,
it will disappear.
You'll be able to run the notebooks,
but the only way to permanently save any work
is to download the files to your machine.
In order to continue from where you left off,
you'd then need to re-upload the files to a new Binder instance.

This option is simple, sufficient for most purposes
(the exercises are very short),
and well-tested.

## Local Install -- Docker

If you'd like to run the materials locally,
the best option is to use
[Docker](https://docs.docker.com/get-docker/),
one of the virtualization technologies
on which Binder is based.

After following the installation instructions for Docker,
build the container with the command
```
docker build -t math-for-ml .
```
and then start it with the command
```
docker run -p 8888:8888 math-for-ml
```
Open a browser window and navigate to
```
localhost:8888
```
and enter, as the password, the token that appears after
`?token` in the URLs printed to the terminal by Jupyter.

## Local Install -- pip/virtualenv

If you are unfamiliar with or unable to use Docker,
you can instead use `pip` to install the necessary packages.
They are located in `requirements-local.txt`
and can be installed with
```
pip install -r requirements-local.txt
```
Note that the requirements are very strictly versioned,
to reduce bugs.
It is highly recommended to use
a virtual environment tool,
like [virtualenv](https://virtualenv.pypa.io/en/latest/)
or [pyenv](https://github.com/pyenv/pyenv)
to set up a specific environment for use with these notebooks.
In general, Python is best used with a virtual environment tool,
so setting one up will brings large dividends for future projects!
