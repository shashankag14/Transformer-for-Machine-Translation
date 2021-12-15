"""
@author : Shashank Agarwal
@when : 11-12-2021
@homepage : https://github.com/shashankag14
"""

import matplotlib.pyplot as plt
from utils import *

def convert_to_list(file):
	with open(file, 'r') as f:
		file_str = f.read()
		file_str = file_str.strip('[]').split(',')
		file_list = [float(val) for val in file_str]
	f.close()
	return file_list

train_loss_file = 'results/train_loss.txt'
valid_loss_file = 'results/valid_loss.txt'
bleu_score_file = 'results/bleu.txt'

def create_plots():
	train_loss = convert_to_list(train_loss_file)
	valid_loss = convert_to_list(valid_loss_file)
	bleu_score = convert_to_list(bleu_score_file)
	num_epochs = len(train_loss)

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	ax1.set_title("Training/Validation Loss vs Epoch")
	ax1.set_xlabel('Epochs')
	ax1.set_ylabel('Loss')

	ax1.plot(range(1,num_epochs+1), train_loss, c='r', label='Training Loss')
	ax1.plot(range(1, num_epochs+1), valid_loss, c='b', label='Validation Loss')

	min_train_loss = min(train_loss)
	min_train_loss_epoch = train_loss.index(min_train_loss)
	plt.axvline(x=min_train_loss_epoch, linestyle='--', color='r', label='Minimum training loss')

	min_valid_loss = min(valid_loss)
	min_valid_loss_epoch = valid_loss.index(min_valid_loss)
	plt.axvline(x=min_valid_loss_epoch, linestyle='--', color = 'b', label='Minimum valid loss')
	ax1.legend()
	plt.grid()
	plt.savefig('results/training_plot.png')
	print("Plot saved in /results/training_plot.png")

if __name__ == '__main__':
	create_plots()