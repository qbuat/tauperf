#!/bin/bash


echo "specific to lxplus"
setupATLAS
lsetup "root 6.04.10-x86_64-slc6-gcc48-opt"

#export DATA_AREA=/eos/atlas/user/q/qbuat/IMAGING
export DATA_AREA=/eos/atlas/user/q/qbuat/IMAGING_bacckup_techlab/IMAGING/
export VE_PATH=/afs/cern.ch/user/q/qbuat/work/public/tau_imaging/imaging_ve

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


# specific to lxplus to avoid https://github.com/Theano/Theano/issues/3780#issuecomment-164843114
export THEANO_FLAGS='gcc.cxxflags="-march=core2"'
