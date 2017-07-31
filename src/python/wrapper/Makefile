all:
	python gen.py
	clang-format --style=Google -i python_wrapper.cc
	g++ -march=native -O3 -shared -std=c++11 -fPIC -I ../../../external/pybind11/include `python-config --cflags --ldflags` python_wrapper.cc -o _falconn.so -I ../../include -I ../../../external/eigen
