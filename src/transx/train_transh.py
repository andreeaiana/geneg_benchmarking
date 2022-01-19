# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/examples/train_transh_FB15K237.py

# import libraries
import os
import numpy as np

# import custom code
from src.OpenKE.openke.config import Trainer
from src.OpenKE.openke.module.model import TransH
from src.OpenKE.openke.module.loss import MarginLoss
from src.OpenKE.openke.module.strategy import NegativeSampling
from src.OpenKE.openke.data import TrainDataLoader
from src.config import DATA_DIR
from src.config import FILENAME_TRANSH_MODEL
from src.config import DKN_KGE_ENTITY_EMBEDDING_DIM

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = str(DATA_DIR) + '/', 
	nbatches = 100,
	threads = 80, 
	sampling_mode = "normal", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# define the model
transh = TransH(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = DKN_KGE_ENTITY_EMBEDDING_DIM, 
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transh, 
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)


# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.5, use_gpu = True)
trainer.run()
transh.save_checkpoint(FILENAME_TRANSH_MODEL)

# save the entity embeddings
embeddings_filepath = os.path.join(DATA_DIR, 'TransH_entity2vec_' + str(DKN_KGE_ENTITY_EMBEDDING_DIM) + '.vec')
results = transh.get_parameters()
entity_embeddings = results['ent_embeddings.weight']
np.savetxt(embeddings_filepath, entity_embeddings)
