.PHONY: attention baseline attention_glove baseline_glove clean printy

printy:
	echo "Specify command please"

attention:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -s train.py -- 'basic_att' --model 'attention' --batch_size 32 --epochs 1 --max_caption_length -1 --workers 32

baseline:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -s train.py -- 'basic_baseline' --model 'baseline' --batch_size 32 --epochs 1 --max_caption_length -1 --workers 32

attention_glove:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -s train.py -- 'glove_att' --model 'attention' --batch_size 32 --epochs 1 --use_glove True --fine_tune_embedding True --embed_size 300 --max_caption_length -1 --workers 32

baseline_glove:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p gpu-shannon -s train.py -- 'glove_baseline' --model 'baseline' --batch_size 32 --epochs 1 --use_glove True --fine_tune_embedding True --embed_size 300 --max_caption_length -1 --workers 32

clean:
	rm slurm-*

