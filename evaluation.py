import numpy as np
from sklearn.model_selection import KFold
from sklearn import svm
import scipy.io as sp

mat35 = sp.loadmat('matlab35.mat')
val35 = np.array(mat35['val'])
raw_data = np.asarray(val35[:,1:]).astype('float64')
raw_target = np.asarray(val35[:,0]).astype('int64')

mat48 = sp.loadmat('matlab48.mat')
val48 = np.array(mat48['dataset2_val'])
mix_data = np.asarray(val48[:,1:]).astype('float64')
mix_target = np.asarray(val48[:,0]).astype('int64')

kf = KFold(n_splits=5,shuffle=True)

scores1 =[]
scores2 =[]

for i in range(10):
	original_scores = []
	for train,test in kf.split(raw_data):
		#print(test)
		train_data = [raw_data[idx] for idx in train]
		train_label = [raw_target[idx] for idx in train]

		clf = svm.SVC(kernel='linear', C=1).fit(train_data, train_label)

		test_data = [raw_data[idx] for idx in test]
		test_label = [raw_target[idx] for idx in test]

		original_scores.append(clf.score(test_data, test_label))

	print()
	original_scores = np.array(original_scores)
	print("Original Data (shuffle)")
	print("-->Accuracy: %0.2f" % (original_scores.mean()))
	scores1.append(original_scores.mean())
	print(">>>>>>>>>>>>>>>")
	mixed_scores = []
	for train,test in kf.split(mix_data):
		#print(test)
		train_data = [mix_data[idx] for idx in train]
		train_label = [mix_target[idx] for idx in train]

		clf = svm.SVC(kernel='linear', C=1).fit(train_data,train_label)

		test_data = [mix_data[idx] for idx in test]
		test_label = [mix_target[idx] for idx in test]

		mixed_scores.append(clf.score(test_data, test_label))

	mixed_scores = np.array(mixed_scores)
	print("Mixed Data (shuffle)")
	print("\t-->Accuracy: %0.2f" % (mixed_scores.mean()))
	scores2.append(mixed_scores.mean())
	print("- - - - - - - - - - - - - - - - - - - - - - - - -")

scores1 = np.array(scores1)
scores2 = np.array(scores2)
print("Original data cv acc mean : "+str(scores1.mean()))
print("Mixed data cv acc mean : "+str(scores2.mean()))

