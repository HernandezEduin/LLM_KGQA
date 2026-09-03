# run twice with same settings and compare outputs
python ./test/qa_subgraph_sampling.py --dataset mquake --hops n --sampling-method random -r --seed 30 > /tmp/neighborhood_1.txt
python ./test/qa_subgraph_sampling.py --dataset mquake --hops n --sampling-method random -r --seed 30 > /tmp/neighborhood_2.txt
diff /tmp/neighborhood_1.txt /tmp/neighborhood_2.txt > /tmp/neighborhood_diff.txt
# if files are exactly the same, diff file will be empty, else it will contain the differences
if [ -s /tmp/neighborhood_diff.txt ]; then
    echo "Reproducibility test failed for random sampling."
    cat /tmp/neighborhood_diff.txt
else
    echo "Reproducibility test passed for random sampling."
fi
# check that the differences above are just ordering differences instead of content differences
#python check_run_files.py
# clean up
rm /tmp/neighborhood_1.txt /tmp/neighborhood_2.txt /tmp/neighborhood_diff.txt