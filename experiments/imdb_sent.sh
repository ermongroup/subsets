#!/bin/sh
export PATH="$PATH:/usr/local/cuda-9.0/bin"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-9.0/lib64"

set -x

pwd; hostname; date

cd ../subsets/L2X/imdb_sent/
mkdir -p data

python explain.py --train --task original

for tau in 0.1 0.5 1.0 2.0 5.0
do
python explain.py --train --task l2x --tau ${tau}
python explain.py --train --task subsets --tau ${tau}
python validate_explanation.py --task l2x --tau ${tau}
python validate_explanation.py --task subsets --tau ${tau}
done

date
