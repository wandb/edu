# adapted from github/BIDS/dats Dockerfile
FROM jupyter/datascience-notebook:04f7f60d34a6

USER $NB_USER

COPY ./requirements.txt ./

# install Python libraries with pip
RUN pip install --no-cache-dir -r requirements.txt

# install Linux tools with apt-get
USER root
RUN apt-get update && apt-get install -y curl ffmpeg graphviz wget

# add files to home directory and rename/reown
COPY ./ /home/$NB_USER/math-for-ml/

RUN usermod -G users $NB_USER && chown -R $NB_USER /home/$NB_USER/ && chgrp -R users /home/$NB_USER/

USER $NB_USER

RUN export USER=$NB_USER
