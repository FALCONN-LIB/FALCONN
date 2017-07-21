all:
	python gen.py
	g++ -march=native -O3 -shared -std=c++11 -fPIC -I ../../../external/pybind11/include `python-config --cflags --ldflags` python_wrapper_new.cc -o falconn_internal.so -I ../../include -I ../../../external/eigen
