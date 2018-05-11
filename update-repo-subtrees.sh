#! /bin/bash
git subtree pull --prefix src/include/falconn/ffht https://github.com/FALCONN-LIB/FFHT.git master --squash
git subtree pull --prefix external/eigen https://github.com/FALCONN-LIB/eigen-mirror.git master --squash
git subtree pull --prefix external/googletest https://github.com/google/googletest.git master --squash
git subtree pull --prefix external/pybind11 https://github.com/pybind/pybind11.git master --squash
git subtree pull --prefix external/simple-serializer https://github.com/ilyaraz/simple-serializer.git master --squash
git subtree pull --prefix external/nlohmann https://github.com/azadkuh/nlohmann_json_release.git master --squash
