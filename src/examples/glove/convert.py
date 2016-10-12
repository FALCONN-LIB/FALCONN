#!/usr/bin/python

import sys
import struct
import numpy as np

matrix = []
with open('dataset/glove.840B.300d.txt', 'r') as inf:
    with open('dataset/glove.840B.300d.dat', 'wb') as ouf:
        counter = 0
        for line in inf:
            row = [float(x) for x in line.split()[1:]]
            assert len(row) == 300
            ouf.write(struct.pack('i', len(row)))
            ouf.write(struct.pack('%sf' % len(row), *row))
            counter += 1
            matrix.append(np.array(row, dtype=np.float32))
            if counter % 10000 == 0:
                sys.stdout.write('%d points processed...\n' % counter)
np.save('dataset/glove.840B.300d', np.array(matrix))
