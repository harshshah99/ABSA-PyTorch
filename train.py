import logging
import argparse
import math
import os
import sys
from time import strftime, localtime
import random
import numpy

from transformers import BertModel

from sklearn import metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, ABSADataset
from atae_lstm import ATAE_LSTM

# from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, LCF_BERT
# from models.aen import CrossEntropyLoss_LSR, AEN_BERT
# from models.bert_spc import BERT_SPC

import matplotlib.pyplot as plt

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
	def __init__(self, opt):
		self.opt = opt

		tokenizer = build_tokenizer(
			fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
			max_seq_len=opt.max_seq_len,
			dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
		embedding_matrix = build_embedding_matrix(
			word2idx=tokenizer.word2idx,
			embed_dim=opt.embed_dim,
			dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
		self.model = opt.model_class(embedding_matrix, opt).to(opt.device)

		self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
		self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)

		if(opt.polarities_dim==2):
			self.trainset = [data_point for data_point in self.trainset if data_point['polarity']!=1]
			for data in self.trainset:
				data['polarity'] = int(data['polarity']/2)
			self.testset = [data_point for data_point in self.testset if data_point['polarity']!=1]
			for data in self.testset:
				data['polarity'] = int(data['polarity']/2)
			

		assert 0 <= opt.valset_ratio < 1
		if opt.valset_ratio > 0:
			valset_len = int(len(self.trainset) * opt.valset_ratio)
			self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
		else:
			self.valset = self.testset

		if opt.device.type == 'cuda':
			logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
		self._print_args()

	def _print_args(self):
		n_trainable_params, n_nontrainable_params = 0, 0
		for p in self.model.parameters():
			n_params = torch.prod(torch.tensor(p.shape))
			if p.requires_grad:
				n_trainable_params += n_params
			else:
				n_nontrainable_params += n_params

		#Adding number of trainable and non trainable parameters to log file
		logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params)) 
		logger.info('> training arguments:')
		for arg in vars(self.opt):
			logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

	def _reset_params(self):
		for child in self.model.children():
			if type(child) != BertModel:  # skip bert params
				for p in child.parameters():
					if p.requires_grad:
						if len(p.shape) > 1:
							self.opt.initializer(p)
						else:
							stdv = 1. / math.sqrt(p.shape[0])
							torch.nn.init.uniform_(p, a=-stdv, b=stdv)

	def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
		max_val_acc = 0 #Maximum Validation accuracy set to 0
		max_val_f1 = 0 #Maximum Validation F1_Score set to 0
		global_step = 0 #Number of steps during training across the tasks set to 0
		path = None

		#Empty lists initialised appending Validation Accuracy, Training Accuracy, Validation F1_Score, Training Loss, Validation Loss
		val_list_acc=[]
		train_list_acc=[]
		val_list_loss=[]
		train_list_loss=[]
		test_loss=[]

		for epoch in range(self.opt.num_epoch): #Loop Continues for Number of Epochs taken as input from Parser 
			logger.info('>' * 100)
			logger.info('epoch: {}'.format(epoch)) #Epoch Number is entered to log file
			n_correct, n_total, loss_total = 0, 0, 0 #total correct outputs, total outputs, total loss is set to 0
			
			# switch model to training mode
			self.model.train()

			#Empty lists initialised appending Batchwise Training Accuracy, Batchwise Training Loss
			train_batchlist_acc=[]
			train_batchlist_loss=[]

			for i_batch, sample_batched in enumerate(train_data_loader): #Batchwise Data is provided to the model for testing
				global_step += 1
				# clear gradient accumulators
				optimizer.zero_grad()

				inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
				outputs = self.model(inputs)
				targets = sample_batched['polarity'].to(self.opt.device)


				print('targets' , targets)
				print('outputs' , outputs)
				loss = criterion(outputs, targets) #Cross Entropy loss is computed
				loss.backward() #computes dloss/dx for every parameter x (Gradient computation)
				optimizer.step() #optimizer.step updates the value of x using the gradient x.grad

				print("PREDICTION _____________________",torch.argmax(outputs, -1))
				print("ACTUAL _________________________",targets)

				n_correct += (torch.argmax(outputs, -1) == targets).sum().item() #Batchwise Count of correct outputs added to total correct outputs
				n_total += len(outputs) #Batchwise Count of outputs added to total outputs
				loss_total += loss.item() * len(outputs) #Batchwise Cross Entropy loss added to total loss
				
				if global_step % self.opt.log_step == 0: #Calculations are done when (Number of steps during training process mod log_step) taken as parser argument is 0
					train_acc = n_correct / n_total #Training Accuracy Calculation
					train_loss = loss_total / n_total #Training Loss Calculation
					train_batchlist_acc.append(train_acc) #Appending Batchwise Training Accuracy
					train_batchlist_loss.append(train_loss) #Appending Batchwise Training Loss
					logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc)) #Batchwise Training Accuracy/Loss entered to log file

			train_list_acc.append(sum(train_batchlist_acc)/len(train_batchlist_acc)) #Appending Average batch training accuracy
			train_list_loss.append(sum(train_batchlist_loss)/len(train_batchlist_loss)) #Appending Average batch training loss
			val_acc, val_f1, val_loss = self._evaluate_acc_f1(val_data_loader,criterion) #Evaluating model on validation dataset
			test_loss.append(val_loss) #Appending validation loss
			val_list_acc.append(val_acc) #Appending validation accuracy
			val_list_loss.append(val_f1) #Appending validation F1_Score
			
			logger.info('> val_acc: {:.4f}, val_f1: {:.4f}, val_loss: {:.4f}'.format(val_acc, val_f1, val_loss))#Validation Accuracy/F1_Score/Loss entered to log file
			
			if val_acc > max_val_acc: #If validation accuracy is greater than maximum validation accuracy model is saved
				max_val_acc = val_acc
				if not os.path.exists('state_dict'):
					os.mkdir('state_dict')
				path = 'state_dict/{0}_{1}_val_acc{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
				torch.save(self.model.state_dict(), path)
				logger.info('>> saved: {}'.format(path))
			
			if val_f1 > max_val_f1: #If validation F1_Score is greater than maximum validation score model update
				max_val_f1 = val_f1
		
		#Saving Plots for Validation Loss, Training Accuracy, Validation Accuracy, Training Loss, Validation F1_Score against Epochs
		plt.plot(test_loss, label='Validation Loss')
		plt.plot(train_list_acc,label='Training Accuracy')
		plt.plot(val_list_acc, label='Validation Accuracy')
		plt.plot(train_list_loss, label='Training Loss')
		plt.plot(val_list_loss, label='Validation F1 Score')
		plt.xlabel('Epoch')
		plt.ylabel('Loss/Accuracy/F1_Score')
		plt.legend()
		plt.savefig('Figure_1.png')
		
		return path

	def _evaluate_acc_f1(self, data_loader, criterion):
		n_correct, n_total, loss_total = 0, 0, 0  #total correct outputs, total outputs, total loss is set to 0
		t_targets_all, t_outputs_all = None, None
		
		# switch model to evaluation mode
		self.model.eval()
		with torch.no_grad(): #turned off gradients computation
			for t_batch, t_sample_batched in enumerate(data_loader): #Batchwise Data is provided to the model for testing
				t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
				t_targets = t_sample_batched['polarity'].to(self.opt.device)
				t_outputs = self.model(t_inputs)
				loss = criterion(t_outputs, t_targets) #Cross Entropy loss is computed

				n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item() #Batchwise Count of correct outputs added to total correct outputs
				n_total += len(t_outputs) #Batchwise Count of outputs added to total outputs
				loss_total += loss.item() * len(t_outputs) #Batchwise Cross Entropy loss added to total loss

				if t_targets_all is None:
					t_targets_all = t_targets
					t_outputs_all = t_outputs
				else:
					t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
					t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

		if(self.opt.polarities_dim==3):
			acc = n_correct / n_total #Accuracy Calculation
			f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro') #F1_Score Calculation
		
		#polarities  = 2
		else: 
			acc = n_correct / n_total #Accuracy Calculation
			f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1], average='macro') #F1_Score Calculation
		
		test_loss = loss_total / n_total #Loss Calculation
		return acc, f1, test_loss

	def run(self):
		"""
		Loading Training/Testing/Validation Dataset->Training Model for given numeber of epochs, Validation of the Model after each epoch (saving the model with maximum validation 
		accuracy)->Saving Progress to log File->Testing the model having the highest Validation accuracy using Testing Dataset->Saving the final outcome to log File
		"""
		# Loss and Optimizer
		criterion = nn.CrossEntropyLoss()
		_params = filter(lambda p: p.requires_grad, self.model.parameters())
		optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

		#Training/Testing/Validation Data Loaded wrt batch sizes defined
		train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
		test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
		val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

		self._reset_params()
		model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader) #Starting the training of Model
		
		self.model.load_state_dict(torch.load(model_path)) # Model with highest validation accuracy loaded for evaluation
		#Model set to evaluation mode
		self.model.eval() 
		test_acc, test_f1, test_loss = self._evaluate_acc_f1(test_data_loader, criterion)
		
		logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}, test_loss: {:.4f}'.format(test_acc, test_f1, test_loss))#Testing Accuracy/F1_Score/Loss entered to log file


def main():
	# Hyper Parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', default='atae_lstm', type=str)
	parser.add_argument('--dataset', default='laptop', type=str, help='twitter, restaurant, laptop')
	parser.add_argument('--optimizer', default='adagrad', type=str, help='adadelta, adagrad, adam, adamax, asgd, rmsprop, sgd')
	parser.add_argument('--initializer', default='xavier_uniform_', type=str, help='xavier_uniform_, xavier_normal_, orthogonal_')
	parser.add_argument('--learning_rate', default=1e-3, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
	parser.add_argument('--dropout', default=0.1, type=float)
	parser.add_argument('--l2reg', default=0.01, type=float)
	parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
	parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
	parser.add_argument('--log_step', default=5, type=int)
	parser.add_argument('--embed_dim', default=300, type=int)
	parser.add_argument('--hidden_dim', default=300, type=int)
	parser.add_argument('--bert_dim', default=768, type=int)
	parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
	parser.add_argument('--max_seq_len', default=80, type=int)
	parser.add_argument('--polarities_dim', default=3, type=int, help='3 for Three Way training (Pos/Neg/Neutral), 2 for Two Way training (Pos/Neg)')
	parser.add_argument('--hops', default=3, type=int)
	parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0 (if GPU is available)')
	parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
	parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
	# # The following parameters are only valid for the lcf-bert model
	# parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
	# parser.add_argument('--SRD', default=3, type=int, help='semantic-relative-distance, see the paper of LCF-BERT model')

	opt = parser.parse_args()

	if opt.seed is not None:
		random.seed(opt.seed)
		numpy.random.seed(opt.seed)
		torch.manual_seed(opt.seed)
		torch.cuda.manual_seed(opt.seed)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False

	#Model Dictionary contatining the list of available models
	model_classes = {'atae_lstm': ATAE_LSTM}

	#Dataset Files Dictionary contating the list of addresses of training/testing according to dataset used (Laptop/Restaurant)
	dataset_files = {
		'twitter': {
			'train': './datasets/acl-14-short-data/train.raw',
			'test': './datasets/acl-14-short-data/test.raw'
		},
		'restaurant': {
			'train': './datasets/semeval14/Restaurants_Train.xml.seg',
			'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg'
		},
		'laptop': {
			'train': './datasets/semeval14/Laptops_Train.xml.seg',
			'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg'
		}
	}
	input_colses = {'atae_lstm': ['text_raw_indices', 'aspect_indices']}

	#Initializers Dictionary containing list of various available initializers
	initializers = {
		'xavier_uniform_': torch.nn.init.xavier_uniform_,
		'xavier_normal_': torch.nn.init.xavier_normal,
		'orthogonal_': torch.nn.init.orthogonal_,
	}

	#Optimizers Dictionary containing list of various available optimizers
	optimizers = {
		'adadelta': torch.optim.Adadelta,  # default lr=1.0
		'adagrad': torch.optim.Adagrad,  # default lr=0.01
		'adam': torch.optim.Adam,  # default lr=0.001
		'adamax': torch.optim.Adamax,  # default lr=0.002
		'asgd': torch.optim.ASGD,  # default lr=0.01
		'rmsprop': torch.optim.RMSprop,  # default lr=0.01
		'sgd': torch.optim.SGD,
	}

	#Parser arguments used to set Model Hyper-parameters
	opt.model_class = model_classes[opt.model_name]
	opt.dataset_file = dataset_files[opt.dataset]
	opt.inputs_cols = input_colses[opt.model_name]
	opt.initializer = initializers[opt.initializer]
	opt.optimizer = optimizers[opt.optimizer]
	opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
		if opt.device is None else torch.device(opt.device)

	#Log File Created for keeping track of the training Process
	log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d-%H%M", localtime()))
	logger.addHandler(logging.FileHandler(log_file))

	ins = Instructor(opt) #Instructor class instance created
	ins.run() #Run function of Instructor class is called


if __name__ == '__main__':
	main()
