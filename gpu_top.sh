

#!/bin/bash

# top like command line tool for GPU utilization

watch "nvidia-smi -q -g 0 -d UTILIZATION | grep 'Gpu\|Memory' | grep -v Samples"

