INC_DIR=src/include/falconn
TEST_DIR=src/test
BENCH_DIR=src/benchmark
PYTHON_DIR=src/python
GTEST_DIR=external/googletest/googletest
GMOCK_DIR=external/googletest/googlemock
TEST_BIN_DIR=test_bin
PYTHON_WRAPPER_DIR=python_wrapper
PYTHON_WRAPPER_FILE=src/python/wrapper/python_wrapper.cc
PYTHON_PKG_DIR=python_package
DOC_DIR=doc

ALL_HEADERS = $(INC_DIR)/core/lsh_table.h $(INC_DIR)/core/cosine_distance.h $(INC_DIR)/core/euclidean_distance.h $(INC_DIR)/core/composite_hash_table.h $(INC_DIR)/core/stl_hash_table.h $(INC_DIR)/core/polytope_hash.h $(INC_DIR)/core/flat_hash_table.h $(INC_DIR)/core/probing_hash_table.h $(INC_DIR)/core/hyperplane_hash.h $(INC_DIR)/core/heap.h $(INC_DIR)/core/prefetchers.h $(INC_DIR)/core/incremental_sorter.h $(INC_DIR)/core/lsh_function_helpers.h $(INC_DIR)/core/hash_table_helpers.h $(INC_DIR)/core/data_storage.h $(INC_DIR)/core/nn_query.h $(INC_DIR)/lsh_nn_table.h $(INC_DIR)/wrapper/cpp_wrapper_impl.h $(INC_DIR)/falconn_global.h $(TEST_DIR)/test_utils.h  $(INC_DIR)/core/data_transformation.h $(INC_DIR)/core/bit_packed_vector.h $(INC_DIR)/core/bit_packed_flat_hash_table.h $(INC_DIR)/core/random_projection_sketches.h $(INC_DIR)/experimental/pipes.h $(INC_DIR)/experimental/code_generation.h

CXX=g++
CXXFLAGS=-std=c++14 -DNDEBUG -Wall -Wextra -march=native -O3 -I external/eigen -I src/include -I external/simple-serializer -I external/nlohmann
NUMPY_INCLUDE_DIR= $(shell python -c "import numpy; print(numpy.get_include())")

clean:
	rm -rf obj
	rm -rf $(TEST_BIN_DIR)
	rm -rf $(DOC_DIR)/html
	rm -rf $(PYTHON_WRAPPER_DIR)
	rm -rf $(PYTHON_PKG_DIR)
	rm -f random_benchmark
	rm -f test-output.txt

docs: $(ALL_HEADERS) $(DOC_DIR)/Doxyfile
	doxygen $(DOC_DIR)/Doxyfile

python_wrapper: $(ALL_HEADERS) $(PYTHON_WRAPPER_FILE)
	rm -rf $(PYTHON_WRAPPER_DIR)
	mkdir -p $(PYTHON_WRAPPER_DIR)
	$(CXX) $(CXXFLAGS) -shared -fPIC -I external/pybind11/include `python-config --cflags --ldflags` $(PYTHON_WRAPPER_FILE) -o $(PYTHON_WRAPPER_DIR)/_falconn.so

python_package:
	rm -rf $(PYTHON_PKG_DIR)
	mkdir -p $(PYTHON_PKG_DIR)
	cp src/python/package/setup.py $(PYTHON_PKG_DIR)
	cp README.md $(PYTHON_PKG_DIR)
	cp LICENSE.txt $(PYTHON_PKG_DIR)
	cp CONTRIBUTORS.md $(PYTHON_PKG_DIR)
	cp src/python/package/MANIFEST.in $(PYTHON_PKG_DIR)
	mkdir -p $(PYTHON_PKG_DIR)/falconn
	cp src/python/package/falconn/__init__.py $(PYTHON_PKG_DIR)/falconn
	cp -r src/include $(PYTHON_PKG_DIR)
	mkdir -p $(PYTHON_PKG_DIR)/internal
	cp src/python/wrapper/python_wrapper.cc $(PYTHON_PKG_DIR)/internal
	mkdir -p $(PYTHON_PKG_DIR)/external
	cp -r external/eigen $(PYTHON_PKG_DIR)/external
	cp -r external/pybind11 $(PYTHON_PKG_DIR)/external
	cp -r external/simple-serializer $(PYTHON_PKG_DIR)/external
	cd $(PYTHON_PKG_DIR); python setup.py sdist; cd dist; tar -xf FALCONN*.tar.gz; cd FALCONN-*; python setup.py build

python_package_install: python_package
	cd $(PYTHON_PKG_DIR)/dist/FALCONN-*; python setup.py install

random_benchmark: $(BENCH_DIR)/random_benchmark.cc $(ALL_HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $(BENCH_DIR)/random_benchmark.cc -pthread

obj/gtest-all.o: $(GTEST_DIR)/src/gtest-all.cc
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -I $(GTEST_DIR) -c -o $@ $<

obj/gtest_main.o: $(GTEST_DIR)/src/gtest_main.cc
	mkdir -p obj
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o $@ $<

$(TEST_BIN_DIR)/cosine_distance_test: $(TEST_DIR)/cosine_distance_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/cosine_distance_test.o $(TEST_DIR)/cosine_distance_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/cosine_distance_test obj/gtest_main.o obj/gtest-all.o obj/cosine_distance_test.o -pthread

$(TEST_BIN_DIR)/euclidean_distance_test: $(TEST_DIR)/euclidean_distance_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/euclidean_distance_test.o $(TEST_DIR)/euclidean_distance_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/euclidean_distance_test obj/gtest_main.o obj/gtest-all.o obj/euclidean_distance_test.o -pthread

$(TEST_BIN_DIR)/flat_hash_table_test: $(TEST_DIR)/flat_hash_table_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/flat_hash_table_test.o $(TEST_DIR)/flat_hash_table_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/flat_hash_table_test obj/gtest_main.o obj/gtest-all.o obj/flat_hash_table_test.o -pthread

$(TEST_BIN_DIR)/probing_hash_table_test: $(TEST_DIR)/probing_hash_table_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/probing_hash_table_test.o $(TEST_DIR)/probing_hash_table_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/probing_hash_table_test obj/gtest_main.o obj/gtest-all.o obj/probing_hash_table_test.o -pthread

$(TEST_BIN_DIR)/stl_hash_table_test: $(TEST_DIR)/stl_hash_table_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/stl_hash_table_test.o $(TEST_DIR)/stl_hash_table_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/stl_hash_table_test obj/gtest_main.o obj/gtest-all.o obj/stl_hash_table_test.o -pthread

$(TEST_BIN_DIR)/composite_hash_table_test: $(TEST_DIR)/composite_hash_table_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/composite_hash_table_test.o $(TEST_DIR)/composite_hash_table_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/composite_hash_table_test obj/gtest_main.o obj/gtest-all.o obj/composite_hash_table_test.o -pthread

$(TEST_BIN_DIR)/polytope_hash_test: $(TEST_DIR)/polytope_hash_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/polytope_hash_test.o $(TEST_DIR)/polytope_hash_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/polytope_hash_test obj/gtest_main.o obj/gtest-all.o obj/polytope_hash_test.o -pthread

$(TEST_BIN_DIR)/hyperplane_hash_test: $(TEST_DIR)/hyperplane_hash_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/hyperplane_hash_test.o $(TEST_DIR)/hyperplane_hash_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/hyperplane_hash_test obj/gtest_main.o obj/gtest-all.o obj/hyperplane_hash_test.o -pthread

$(TEST_BIN_DIR)/lsh_table_test: $(TEST_DIR)/lsh_table_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/lsh_table_test.o $(TEST_DIR)/lsh_table_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/lsh_table_test obj/gtest_main.o obj/gtest-all.o obj/lsh_table_test.o -pthread

$(TEST_BIN_DIR)/heap_test: $(TEST_DIR)/heap_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/heap_test.o $(TEST_DIR)/heap_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/heap_test obj/gtest_main.o obj/gtest-all.o obj/heap_test.o -pthread

$(TEST_BIN_DIR)/incremental_sorter_test: $(TEST_DIR)/incremental_sorter_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/incremental_sorter_test.o $(TEST_DIR)/incremental_sorter_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/incremental_sorter_test obj/gtest_main.o obj/gtest-all.o obj/incremental_sorter_test.o -pthread

$(TEST_BIN_DIR)/nn_query_test: $(TEST_DIR)/nn_query_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/nn_query_test.o $(TEST_DIR)/nn_query_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/nn_query_test obj/gtest_main.o obj/gtest-all.o obj/nn_query_test.o -pthread

$(TEST_BIN_DIR)/cpp_wrapper_test: $(TEST_DIR)/cpp_wrapper_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/cpp_wrapper_test.o $(TEST_DIR)/cpp_wrapper_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/cpp_wrapper_test obj/gtest_main.o obj/gtest-all.o obj/cpp_wrapper_test.o -pthread

$(TEST_BIN_DIR)/data_transformation_test: $(TEST_DIR)/data_transformation_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/data_transformation_test.o $(TEST_DIR)/data_transformation_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/data_transformation_test obj/gtest_main.o obj/gtest-all.o obj/data_transformation_test.o -pthread

$(TEST_BIN_DIR)/data_storage_test: $(TEST_DIR)/data_storage_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/data_storage_test.o $(TEST_DIR)/data_storage_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/data_storage_test obj/gtest_main.o obj/gtest-all.o obj/data_storage_test.o -pthread

$(TEST_BIN_DIR)/bit_packed_vector_test: $(TEST_DIR)/bit_packed_vector_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/bit_packed_vector_test.o $(TEST_DIR)/bit_packed_vector_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/bit_packed_vector_test obj/gtest_main.o obj/gtest-all.o obj/bit_packed_vector_test.o -pthread

$(TEST_BIN_DIR)/bit_packed_flat_hash_table_test: $(TEST_DIR)/bit_packed_flat_hash_table_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/bit_packed_flat_hash_table_test.o $(TEST_DIR)/bit_packed_flat_hash_table_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/bit_packed_flat_hash_table_test obj/gtest_main.o obj/gtest-all.o obj/bit_packed_flat_hash_table_test.o -pthread

$(TEST_BIN_DIR)/sketches_test: $(TEST_DIR)/sketches_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -c -o obj/sketches_test.o $(TEST_DIR)/sketches_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/sketches_test obj/gtest_main.o obj/gtest-all.o obj/sketches_test.o -pthread

$(TEST_BIN_DIR)/pipes_test: $(TEST_DIR)/pipes_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -I $(GMOCK_DIR)/include -c -o obj/pipes_test.o $(TEST_DIR)/pipes_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/pipes_test obj/gtest_main.o obj/gtest-all.o obj/pipes_test.o -pthread

$(TEST_BIN_DIR)/pipe_generation_test: $(TEST_DIR)/pipe_generation_test.cc $(ALL_HEADERS) obj/gtest-all.o obj/gtest_main.o
	mkdir -p $(TEST_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I $(GTEST_DIR)/include -I $(GMOCK_DIR)/include -c -o obj/pipe_generation_test.o $(TEST_DIR)/pipe_generation_test.cc
	$(CXX) $(CXXFLAGS) -o $(TEST_BIN_DIR)/pipe_generation_test obj/gtest_main.o obj/gtest-all.o obj/pipe_generation_test.o -pthread

run_all_cpp_tests: $(TEST_BIN_DIR)/bit_packed_flat_hash_table_test $(TEST_BIN_DIR)/bit_packed_vector_test $(TEST_BIN_DIR)/composite_hash_table_test $(TEST_BIN_DIR)/cosine_distance_test $(TEST_BIN_DIR)/cpp_wrapper_test $(TEST_BIN_DIR)/data_storage_test $(TEST_BIN_DIR)/data_transformation_test $(TEST_BIN_DIR)/euclidean_distance_test $(TEST_BIN_DIR)/flat_hash_table_test $(TEST_BIN_DIR)/heap_test $(TEST_BIN_DIR)/hyperplane_hash_test $(TEST_BIN_DIR)/incremental_sorter_test $(TEST_BIN_DIR)/lsh_table_test $(TEST_BIN_DIR)/nn_query_test $(TEST_BIN_DIR)/pipe_generation_test $(TEST_BIN_DIR)/pipes_test $(TEST_BIN_DIR)/polytope_hash_test $(TEST_BIN_DIR)/probing_hash_table_test $(TEST_BIN_DIR)/sketches_test $(TEST_BIN_DIR)/stl_hash_table_test
	./$(TEST_BIN_DIR)/bit_packed_flat_hash_table_test
	./$(TEST_BIN_DIR)/bit_packed_vector_test
	./$(TEST_BIN_DIR)/composite_hash_table_test
	./$(TEST_BIN_DIR)/cosine_distance_test
	./$(TEST_BIN_DIR)/cpp_wrapper_test
	./$(TEST_BIN_DIR)/data_storage_test
	./$(TEST_BIN_DIR)/data_transformation_test
	./$(TEST_BIN_DIR)/euclidean_distance_test
	./$(TEST_BIN_DIR)/flat_hash_table_test
	./$(TEST_BIN_DIR)/heap_test
	./$(TEST_BIN_DIR)/hyperplane_hash_test
	./$(TEST_BIN_DIR)/incremental_sorter_test
	./$(TEST_BIN_DIR)/lsh_table_test
	./$(TEST_BIN_DIR)/nn_query_test
	./$(TEST_BIN_DIR)/pipe_generation_test
	./$(TEST_BIN_DIR)/pipes_test
	./$(TEST_BIN_DIR)/polytope_hash_test
	./$(TEST_BIN_DIR)/probing_hash_table_test
	./$(TEST_BIN_DIR)/sketches_test
	./$(TEST_BIN_DIR)/stl_hash_table_test

run_all_python_tests:
	py.test src/python/test
