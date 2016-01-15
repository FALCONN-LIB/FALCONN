#!/bin/bash

cd /var/www/html/falconn/pdoc
rm -rf falconn
pdoc --html falconn
