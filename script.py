import numpy as np 
from l2_distance import l2_distance
from run_knn import run_knn


training_data_set = np.load('mnist_train.npz')
train_data = training_data_set['train_inputs']
train_label = training_data_set['train_targets']
valid_data_set = np.load('mnist_valid.npz')
valid_data = valid_data_set['valid_inputs'] 
valid_label = valid_data_set['valid_targets'] 
k = 9




valid_labels_knn = run_knn(k, train_data, train_label, valid_data)
print valid_labels_knn


num_correct_prediction = 0 
num_total_points = 0 
classification_rate = 0

count = 0
correct_count = 0
for valid_label_knn in valid_labels_knn:
	if valid_label_knn == valid_label[count]:
		correct_count = correct_count + 1
	count = count + 1

print correct_count
