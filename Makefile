.PHONY: attention baseline attention_glove baseline_glove attention_bert clean printy

printy:
	echo "Specify command please"

attention:
	export NLTK_DATA='nltk_data' ; bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu-markov -c 80 -s train.py -- 'basic_att' --model 'attention' --batch_size 16 --epochs 1 --max_caption_length -1 --workers 80

baseline:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -s train.py -- 'basic_baseline' --model 'baseline' --batch_size 32 --epochs 1 --max_caption_length -1 --workers 32

attention_glove:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 32 -s train.py -- 'glove_att' --model 'attention' --batch_size 32 --epochs 4 --use_glove True --fine_tune_embedding True --embed_size 300 --checkpoint 'glove_att_1.pth.tar' --max_caption_length -1 --workers 32

baseline_glove:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 32 -s train.py -- 'glove_baseline' --model 'baseline' --batch_size 32 --epochs 1 --use_glove True --fine_tune_embedding True --embed_size 300 --max_caption_length -1 --workers 32

attention_bert:
	bash /opt/local/bin/run_py_job.sh -e dit245_group15 -p cpu -c 32 -s train.py -- 'bert_attention' --model 'attention' --batch_size 32 --epochs 4 --use_bert True --fine_tune_embedding True --embed_size 768 --checkpoint 'bert_attention_2.pth.tar' --max_caption_length -1 --workers 32
clean:
	rm slurm-*

