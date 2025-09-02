#!/usr/bin/env bash

find . -iname '*.cc' -o -iname '*.h' | xargs clang-format -i

black .

python3 ./update_header_hashes.py


