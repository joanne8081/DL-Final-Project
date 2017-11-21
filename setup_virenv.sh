##!/usr/bin/env bash

#sudo pip install virtualenv  
#virtualenv .env                  # Create a virtual environment
#source .env/bin/activate         # Activate the virtual environment
#sudo apt-get install libcupti-dev
#pip install --upgrade tensorflow-gpu  # for Python 2.7 with GPU
#pip install tflearn
#pip install opencv-python
#sudo apt-get build-dep python-imaging
#sudo apt-get install libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
#pip install tensorflow            # for Python 2.7 with GPU
#pip install -r requirements.txt  # Install dependencies
#deactivate
#echo "**************************************************"
#echo "***  Successfully Activate Virtual Environment ***"
#echo "**************************************************"

#my file: /home/james23468/PointSetGeneration/depthestimate/ptsft76.tar.gz

wget http://repo.continuum.io/archive/Anaconda2-5.0.1-Linux-x86_64.sh
bash Anaconda2-5.0.1-Linux-x86_64.sh
export PATH=“/home/james23468/anaconda2/bin:$PATH”
source ~/.bashrc
conda create --name venv python=2.7 anaconda
source activate venv
pip install Cython Jinja2 MarkupSafe opencv-python Pillow Pygments appnope argparse backports-abc backports.ssl-match-hostname certifi cycler decorator future gnureadline h5py ipykernel ipython ipython-genutils ipywidgets jsonschema jupyter jupyter-client jupyter-console jupyter-core matplotlib mistune nbconvert nbformat nltk notebook numpy path.py pexpect pickleshare ptyprocess pyparsing python-dateutil pytz pyzmq qtconsole scipy simplegeneric singledispatch site six tensorflow tensorflow-gpu terminado tflearn tornado traitlets
