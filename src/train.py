import torch
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss

import os
import time 


class Train(object):
	'''----------------------------------------------------------------
	Initialize train object, check/create final_model and training_models.
	Load configuration and model from the given path. Create optimizer, 
	scheduler and loss_function.
	Args:
		device		 	: torch_device
		model_path	 	: str
		tokenizer_len	: int
		ignore_index 	: int
		train_loader_len: int
	Return: 
		object
	'''
	def __init__(self, device, model_path, tokenizer_len, ignore_index, train_loader_len, config):

		self.device = device
		self.config = config
	
		self.final_model = os.path.join(config.out_dir, config.final_model)
		if not os.path.exists(self.final_model):
			os.makedirs(self.final_model)

		self.training_models = os.path.join(config.out_dir, config.training_models)
		if not os.path.exists(self.training_models):
			os.makedirs(self.training_models)
		
		self.configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=False), out_dir=config["out_dir"]) # added
		self.model = GPT2LMHeadModel.from_pretrained(model_path,  config=self.configuration)
		self.model.resize_token_embeddings(tokenizer_len)
		self.model.to(self.device)
		
		self.optimizer = AdamW(self.model.parameters(), lr= 5e-4, eps= 1e-8)
		self.loss_fct = CrossEntropyLoss(ignore_index=ignore_index) #ignores padding token for loss calculation

		self.gradient_accumulation_steps = 32
		self.max_grad_norm = 1

		# Create the learning rate scheduler. This changes the learning rate as the training loop progresses
		self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps =  1e2, num_training_steps = train_loader_len * 50)


	'''----------------------------------------------------------------
	Takes a batch to process for training loop.
	Args:
		batch: tensors
	Return: 
		shift_logits, shift_labels: tensors
	'''
	def process_train_batch(self, batch):   
		# TODO: Complete this function      
		inputs, labels = batch['text'].clone().detach(), batch['text'].clone().detach()
		inputs = inputs.to(self.device)
		labels = labels.to(self.device)

		logits = self.model(inputs)[0]
		
		idx = batch[] # index of separator token
		#print('logits: {}'.format(logits.shape))
		#print('idx: {}'.format(idx))
		# only consider loss on reference summary just like seq2seq models
		shift_logits = logits[..., idx:-1, :].contiguous()
		shift_labels = labels[..., idx+1:].contiguous()

		return shift_logits, shift_labels 


	def process_eval_batch(self, batch):
		inputs, labels = batch['text'].clone().detach(), batch['text'].clone().detach()
		inputs = inputs.to(self.device)
		labels = labels.to(self.device)
	
		with torch.no_grad():        
			logits = self.model(inputs)[0]
			#print('logits: {}'.format(logits.shape))
			
			idx = batch['s_idx'].item() 
			shift_logits = logits[..., idx:-1, :].contiguous()
			shift_labels = labels[..., idx+1:].contiguous()

		return shift_logits, shift_labels


	def train_loop(self, train_dataloader):
		# TODO: Complete training loop
		self.model.train()
		total_train_loss = 0

		for step, batch in enumerate(train_dataloader):
			            self.optimizer.zero_grad()  # put all gradients to zero 
            shift_logits, shift_labels =  self.process_train_batch(batch)# prepare batch data

			loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
			loss = loss/self.gradient_accumulation_steps
			loss.backward() 
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)       
			self.optimizer.step()
			self.scheduler.step()
			batch_loss = loss.item()
			total_train_loss += batch_loss

		return total_train_loss

	def eval_loop(self, val_dataloader):
		self.model.eval()
		total_eval_loss = 0
		
		for step, batch in enumerate(val_dataloader):
			shift_logits, shift_labels = self.process_eval_batch(batch)	

			loss = self.loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
			batch_loss = loss.item()
			total_eval_loss += batch_loss

		return total_eval_loss


	def average_loss(self, total_loss, size):
		return total_loss / size


	def save_model(self, path):
		model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Take care of distributed/parallel training
		model_to_save.save_pretrained(path)

	def model_params(self):

		name = "params_" + str(int(time.time())) + ".txt"
		file = os.path.join(self.config.out_dir, name)
		# Get all of the model's parameters as a list of tuples.
		params = list(self.model.named_parameters())
		f = open(file, "w")

		f.write('The model has {:} different named parameters.\n'.format(len(params)))
		f.write('\n==== Embedding Layer ====\n')
		for p in params[0:2]:
			f.write("{:<55} {:>12}\n".format(p[0], str(tuple(p[1].size()))))

		f.write('\n==== First Transformer ====\n')
		for p in params[2:14]:
			f.write("{:<55} {:>12}\n".format(p[0], str(tuple(p[1].size()))))

		f.write('\n==== Output Layer ====\n')
		for p in params[-2:]:
			f.write("{:<55} {:>12}\n".format(p[0], str(tuple(p[1].size()))))	
		f.close()

	def train_model(self, train_dataloader, val_dataloader):
		'''----------------------------------------------------------------
		'''
		epochs = 50
		for epoch_i in range(0, epochs):

			print('======== Epoch {}/{} ========'.format(epoch_i + 1, epochs))
			#print('Training...')

			total_train_loss = self.train_loop(train_dataloader)
			avg_train_loss = self.average_loss(total_train_loss, len(train_dataloader))     
			
			print("  Average training loss: {0:.2f}".format(avg_train_loss))
						
			#print("Running Validation...")

			total_eval_loss = self.eval_loop(val_dataloader)   
			avg_val_loss = self.average_loss(total_eval_loss, len(val_dataloader)) 

			print("  Valid. Loss: {0:.2f}".format(avg_val_loss))

			if (epoch_i + 1)% 10 == 0: 
				#save model
				print('epoch_i + 1: {}'.format(epoch_i + 1))
				path = os.path.join(self.training_models, str(epoch_i+1))
				self.save_model(self.training_models)

		print("Training complete!")
		

		self.save_model(self.final_model)
		
