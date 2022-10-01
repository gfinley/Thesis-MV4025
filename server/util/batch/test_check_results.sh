sort results.csv > results_sorted.csv
sort test_results_gold.csv > test_results_gold_sorted.csv
if  diff results_sorted.csv test_results_gold_sorted.csv
then
    echo "Test passed"
else
    echo "Test failed"
fi
rm results_sorted.csv test_results_gold_sorted.csv