
.PHONY: shannon clean

train_shannon:
	bash /opt/local/bin/run_py_job.sh -s train.py -e dit245_group15 -p gpu-shannon

clean:
	rm slurm-*
