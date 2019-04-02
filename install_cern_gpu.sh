
source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
lsetup root
lsetup cmake

cd /data/${USER}

echo 'create the virtual environement'
wget https://pypi.python.org/packages/source/v/virtualenv/virtualenv-15.2.0.tar.gz
tar xvfz virtualenv-15.2.0.tar.gz
python virtualenv-15.2.0/virtualenv.py imaging_ve_gpu
source imaging_ve_gpu/bin/activate

echo 'install python packages'
pip install pip --upgrade
pip install tensorflow
pip install tensorflow-gpu
pip install theano
pip install keras
pip install pydot_ng
pip install h5py
pip install tables
pip install scikit-learn
pip install scikit-image
pip install matplotlib
pip install root_numpy
pip install rootpy
pip install tabulate
pip install cython
pip install nose
pip install seaborn

# pip install theano==0.9.0
# pip install keras==2.0.6
# pip install pydot_ng==1.0.0
# pip install h5py==2.6.0
# pip install tables==3.3.0
# pip install scikit-learn==0.19.0
# pip install scikit-image==0.12.3
# pip install matplotlib==1.5.3
# pip install root_numpy==4.5.2
# pip install rootpy==0.8.3
# pip install tabulate==0.7.5
# pip install cython
# pip install nose
# pip install tensorflow
