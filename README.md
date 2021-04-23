# WiNR
a prototype verification tool
## Requirements
Gurobi's Python interface, python3.6 or higher, 
tensorflow 1.12 or higher, numpy.

## Installation
Clone the repository via git as follows:
```
git clone 
```

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

## Usage
```
python main.py
```
