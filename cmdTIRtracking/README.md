1. Put the model into cross-modal distialltion/pytracking/networks/cmd35.pth.tar

2. #### Setup the environment  
Create the default environment setting files. 
```bash
# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  

3. train:
	python run_training.py dimp super_dimp

4. test: 
	python run_tracker.py --dataset lsotb dimp super_dimp
	Here, you need to change the pretrainmodel used in ....../ltr/models/tracking/dimpnet.py.
