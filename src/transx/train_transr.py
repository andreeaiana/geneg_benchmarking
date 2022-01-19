# -*- coding: utf-8 -*-

# DISCLAIMER
# This code file is forked and adapted from https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/examples/train_transr_FB15K237.py

# import libraries
import os
import numpy as np

# import custom code
from src.OpenKE.openke.config import Trainer
from src.OpenKE.openke.module.model import TransE, TransR
from src.OpenKE.openke.module.loss import MarginLoss
from src.OpenKE.openke.module.strategy import NegativeSampling
from src.OpenKE.openke.data import TrainDataLoader
from src.config import DATA_DIR
from src.config import FILENAME_TRANSR_MODEL, FILENAME_TRANSR_TRANSE_MODEL
from src.config import DKN_KGE_ENTITY_EMBEDDING_DIM


# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = str(DATA_DIR) + '/', 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0)

# define the model
transe = TransE(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = DKN_KGE_ENTITY_EMBEDDING_DIM, 
	p_norm = 1, 
	norm_flag = True)

model_e = NegativeSampling(
	model = transe, 
	loss = MarginLoss(margin = 5.0),
	batch_size = train_dataloader.get_batch_size())

transr = TransR(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = DKN_KGE_ENTITY_EMBEDDING_DIM,
	dim_r = DKN_KGE_ENTITY_EMBEDDING_DIM,
	p_norm = 1, 
	norm_flag = True,
	rand_init = False)

model_r = NegativeSampling(
	model = transr,
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)

# pretrain transe
trainer = Trainer(model = model_e, data_loader = train_dataloader, train_times = 1, alpha = 0.5, use_gpu = True)
trainer.run()
parameters = transe.get_parameters()
transe.save_parameters(FILENAME_TRANSR_TRANSE_MODEL)

# train transr
transr.set_parameters(parameters)
trainer = Trainer(model = model_r, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transr.save_checkpoint(FILENAME_TRANSR_MODEL)

# save the entity embeddings
embeddings_filepath = os.path.join(DATA_DIR, 'TransR_entity2vec_' + str(DKN_KGE_ENTITY_EMBEDDING_DIM) + '.vec')
results = transr.get_parameters()
entity_embeddings = results['ent_embeddings.weight']
np.savetxt(embeddings_filepath, entity_embeddings)
