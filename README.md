# environmentSim
Environment simulation ROS node for the RACECAR platform. Uses a neural network to simulate racecar dynamics, and rangelibc to simluate laser scans.

A video of the simulator in action is available [here](https://www.youtube.com/watch?v=P9lvrE0fSrY).

To install [range_libc](https://github.com/kctess5/range_libc):

```
pip install --user cython
git clone http://github.com/kctess5/range_libc
cd range_libc/pywrappers
# on VM
./compile.sh
# on car - compiles GPU ray casting methods
./compile_with_cuda.sh
```
