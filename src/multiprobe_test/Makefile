all:
	g++ wrapper.cpp -o _multiprobe.so -O3 -shared -fPIC -Wall -Wno-int-in-bool-context -march=native -I ../include/ -I ../../external/eigen/ -I ../../external/simple-serializer/ -I ../../external/pybind11/include `python3-config --cflags --ldflags`
