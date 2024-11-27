# Blocksworld
## Run

We provide the scripts to reproduce the results of RAP with llama3.1-8b-Instruct. 

```bash
./examples/RAP/blocksworld/test_rap.sh
```

After the run, you can use `aggregate.py` to calculate an overall accuracy of all subsets.

If you want to modify the experiment settings or develop your own method, you may look at `rap_inference.py`.
