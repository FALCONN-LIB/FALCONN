#! /bin/bash

FILENAME="test-output.txt"

make clean 2>&1 | tee $FILENAME
make run_all_cpp_tests 2>&1 | tee -a $FILENAME
printf "\n\n=========================================================================\n\n\n" | tee -a $FILENAME
make random_benchmark 2>&1 | tee -a $FILENAME
./random_benchmark 2>&1 | tee -a $FILENAME
