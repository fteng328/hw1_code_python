import numpy as np 
from l2_distance import l2_distance
from run_knn import run_knn


training_data_set = np.load('mnist_train.npz')
train_data = training_data_set['train_inputs']
train_label = training_data_set['train_targets']
valid_data_set = np.load('mnist_valid.npz')
valid_data = valid_data_set['valid_inputs'] 
valid_label = valid_data_set['valid_targets'] 
test_set = np.load('mnist_test.npz')
test_data = test_set['test_inputs']
test_label = test_set['test_targets']
k = 5




test_labels_knn = run_knn(k, train_data, train_label, test_data)
print test_labels_knn


num_correct_prediction = 0 
num_total_points = 0 
classification_rate = 0

count = 0
correct_count = 0
for test_label_knn in test_labels_knn:
	if test_label_knn == test_label[count]:
		correct_count = correct_count + 1
	count = count + 1

print correct_count
