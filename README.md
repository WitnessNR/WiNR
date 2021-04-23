# WiNR
a prototype verification tool
## Requirements
Gurobi's Python interface, python3.6 or higher, 
tensorflow 1.12 or higher, numpy.

## Installation
Clone the repository via git as follows:
```
git clone https://github.com/WitnessNR/WiNR.git
cd WiNR
```
Then install Gurobi:
```
wget https://packages.gurobi.com/9.0/gurobi9.0.0_linux64.tar.gz
tar -xvf gurobi9.0.0_linux64.tar.gz
cd gurobi900/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
cp lib/libgurobi90.so /usr/local/lib
python3 setup.py install
cd ../../
```
Update environment variables:
```
export GUROBI_HOME="$PWD/gurobi900/linux64"
export PATH="$PATH:${GUROBI_HOME}/bin"
export CPATH="$CPATH:${GUROBI_HOME}/include"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib:${GUROBI_HOME}/lib
```
Note that to run with Gurobi one needs to obtain an academic license for gurobi from
https://user.gurobi.com/download/licenses/free-academic.

If gurobipy is not found despite executing ```python setup.py install``` in the corresponding gurobi directory, gurobipy can alternatively be installed using conda with
```
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
```

In addition, one needs to download GTSRB database before running models trained on GTSRB database. The download link of GTSRB database: https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html.

## Usage
```
cd WiNR
python main.py
```
