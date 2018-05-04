#!/bin/bash

function download() {
    echo "try to get the version: ${1} ..."
    rm json.hpp
    wget -c "https://github.com/nlohmann/json/releases/download/v${1}/json.hpp"
}

[[ -n "$1" ]] && download ${1} || echo -e "syntax: ${0} version\nsample: $> ${0} 2.1.2"
