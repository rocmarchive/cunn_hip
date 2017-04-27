Install Notes
=============
Install on Ubuntu or OSX
------------------------

* Install torch (Refer the repos wiki for installation of the same)

* Then install the rest of the needed packages:
```
luarocks install cutorch
luarocks install cunn
luarocks install csvigo
luarocks install nnx
luarocks install torchffi
luarocks install env
luarocks install graphicsmagick
apt-get install libgraphicsmagick-dev
```

* Download the data from: http://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/data
* Place the data files into a subfolder "data".
* Then run the script dataprep.sh
