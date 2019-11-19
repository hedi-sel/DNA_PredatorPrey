


Requirements
====================

 * CMake (3.15+), g++-7 (C++ 11 standard)
 * Boost Library (1_70+ recommended)
 * CUDA Compatible Graphics card, adn CUDA toolkit (7.0). Tested with Ubuntu 18 on the Nvidia Titan V
 
 <h4>Optional</h4>
 * Python 3.6, with the following modules: 
    * Numpy 
    * Matplotlib
    * OS
 * Imagick package (sudo apt-get install imagemagick)

Building
====================

<h3>Linux</h3>

```c++
./bootstrap.sh
cmake ./    
make
```

Run
====================

run/predatorPrey

Plot results
====================

To plot the results as a series of graphs, in the plot/ subdirectory:

```c++
python3 Ploter.py
```

To make the plotted graphs into a more explicit video

```c++
convert -delay 10 -loop 0 $(ls -1 plot/*.png | sort -V) clip.gif
```