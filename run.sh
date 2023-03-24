test_case=$1
ml_sa_focus=${2-VaR}
(cd src && make)
bin/main --test_case $test_case --ml_sa_focus $ml_sa_focus | tee out/test_case_$test_case.csv
python3 script/process.py --test_case $test_case --ml_sa_focus $ml_sa_focus out/test_case_$test_case.csv fig/test_case_$test_case
for file in fig/test_case_$test_case/*.pdf; do xdg-open $file; done
