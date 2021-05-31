rm -rf saved_models
mkdir saved_models

# Uninstall mkl for faster neural-network training time
pip uninstall -y mkl
# Upgrade pip to ensure the latest package versions are available
python3 -m pip uninstall autogluon.core --y
python3 -m pip uninstall autogluon --y
python3 -m pip uninstall gluoncv --y
pip uninstall dask --y
python3 -m pip install "gluoncv==0.9.0"
# Upgrade setuptools to be compatible with namespace packages
pip install -U setuptools
python3 -m pip install "mxnet<2.0.0, >=1.7.0"
#Install pre-release, frozen to a particual pre-release for stability
pip install --pre "autogluon==0.0.16b20201214"