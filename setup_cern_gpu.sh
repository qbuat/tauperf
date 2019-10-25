#!/bin/bash


echo "specific to lxplus"
setupATLAS
lsetup "root 6.04.10-x86_64-slc6-gcc48-opt"
lsetup hdf5
export HDF5_DIR=$ALRB_HDF5_PATH

#export DATA_AREA=/data/vol/qbuat/IMAGING
#export DATA_AREA=/eos/atlas/user/q/qbuat/IMAGING_bacckup_techlab/IMAGING/
export DATA_AREA=/data/qbuat/data/
export VE_PATH=/data/qbuat/imaging_ve_gpu

echo 'activating the virtual environment'
source ${VE_PATH}/bin/activate



SOURCE_TAUPERF_SETUP="${BASH_SOURCE[0]:-$0}"
DIR_TAUPERF_SETUP="$( dirname "$SOURCE_TAUPERF_SETUP" )"
while [ -h "$SOURCE_TAUPERF_SETUP" ]
do 
  SOURCE_TAUPERF_SETUP="$(readlink "$SOURCE_TAUPERF_SETUP")"
  [[ $SOURCE_TAUPERF_SETUP != /* ]] && SOURCE_TAUPERF_SETUP="$DIR_TAUPERF_SETUP/$SOURCE_TAUPERF_SETUP"
  DIR_TAUPERF_SETUP="$( cd -P "$( dirname "$SOURCE_TAUPERF_SETUP"  )" && pwd )"
  echo $SOURCE_TAUPERF_SETUP
  echo $DIR_TAUPERF_SETUP
done
DIR_TAUPERF_SETUP="$( cd -P "$( dirname "$SOURCE_TAUPERF_SETUP" )" && pwd )"

echo $DIR_TAUPERF_SETUP
echo "sourcing ${SOURCE_TAUPERF_SETUP}..."

export PATH=${DIR_TAUPERF_SETUP}${PATH:+:$PATH}
export PYTHONPATH=${DIR_TAUPERF_SETUP}${PYTHONPATH:+:$PYTHONPATH}
export CPATH=$CPATH:~/.local/include
export LIBRARY_PATH=$LIBRARY_PATH:~/.local/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib

#TensorFlow Fix
export PATH=${PATH}:/usr/local/cuda-9.0/bin
export CUDA_HOME=${CUDA_HOME}:/usr/local/cuda:/usr/local/cuda-9.0
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda-9.0/lib64
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

# specific to lxplus to avoid https://github.com/Theano/Theano/issues/3780#issuecomment-164843114
export THEANO_FLAGS='floatX=float32 device=cpu cast_policy=numpy+floatX'
#export THEANO_FLAGS='gcc.cxxflags="-march=core2"'
