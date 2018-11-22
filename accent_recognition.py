from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
import scipy.io.wavfile as wav
import scipy.interpolate as interpol
import glob
import wave
import pickle
import sklearn
import scipy.io.wavfile as wav
import csv
import os
import array
import re
import numpy as np
import itertools
from time import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import libsvm
import subprocess
from sklearn import metrics
from subprocess import Popen
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import average_precision_score


gnuplot_exe = r"C:\Users\Takudzwa Raisi\Desktop\accent_recognition\gnuplot\binary\gnuplot.exe"
grid_py = r"C:\Users\Takudzwa Raisi\Desktop\accent_recognition\libsvm-3.22\tools\grid.py"
svmtrain_exe = r"C:\Users\Takudzwa Raisi\Desktop\accent_recognition\libsvm-3.17-GPU_x64-v1.2\windows\svm-train-gpu.exe"
svmpredict_exe = r"C:\Users\Takudzwa Raisi\Desktop\accent_recognition\libsvm-3.17-GPU_x64-v1.2\windows\svm-predict.exe"

def paramsfromexternalgridsearch(filename, crange, grange, printlines=False):
	#printlines specifies whether or not the function should print every line of the grid search verbosely
	cmd = 'python "{0}" -log2c {1} -log2g {2} -svmtrain "{3}" -gnuplot "{4}" "{5}"'.format(grid_py, crange, grange, svmtrain_exe, gnuplot_exe, filename)
	f = Popen(cmd, shell = True, stdout = subprocess.PIPE).stdout

	line = ''
	while True:
		last_line = line
		line = f.readline()
		if not line: break
		if printlines: print(line)
	c,g,rate = map(float,last_line.split())
	return c,g,rate

def accuracyfromexternalpredict(scaled_test_file, model_file, predict_test_file, predict_output_file):
	cmd = '"{0}" "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
	f = Popen(cmd, shell = True, stdout = subprocess.PIPE).stdout
	#f = subprocess.Popen(cmd, shell = True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)

	line = ''
	while True:
		last_line = line
		line = f.readline()
		if not line: break

	return last_line.split(" ")[3][1:-1].split("/")[0], last_line.split(" ")[3][1:-1].split("/")[1]

def normalize(inSig,outLen):
	#This function normalizes the audio signal.
	#It first produces an interp1d structure that readily interpolates between points
	#Then it sets the size of the space to outLen=200000 points, and interp1d interpolates to fill in gaps
	#In essence, it takes every audio signal and produces a signal with outLen=200000 data points in it = normalization
    inSig = np.array(inSig)
    arrInterpol = interpol.interp1d(np.arange(inSig.size),inSig)
    arrOut = arrInterpol(np.linspace(0,inSig.size-1,outLen))
    return arrOut


def writetopcklfile(outpath, data):
	with open(outpath, 'wb') as f:
		pickle.dump(data, f)

def readfrompcklfile(outpath):
		with open(outpath, 'rb') as f:
			return pickle.load(f)

def custom_dump_svmlight_file(X_train,Y_train,filename):
    #This function inserts the extracted features in the libsvm format
    featinds = [" " + str(i) + ":" for i in range(1, len(X_train[0])+1)]
    with open(filename, 'w') as f:
        for ind, row in enumerate(X_train):
            f.write(str(Y_train[ind]) + " " + "".join([x for x in itertools.chain.from_iterable(zip(featinds,map(str,row))) if x]) + "\n")

def main():

	start = time()

	path =r'accents'
	files = os.listdir(path)

	features = []
	label = []
	Filenames = {}
	X =[]
	Xdir = {}
	y =[]	
	for filename in glob.glob(os.path.join(path, '*.wav')):
		Filenames[filename[filename.find('\\')+1:]] = filename
			
		
	for sounddata in Filenames:
		(rate,sig) =  wav.read(path+'/'+sounddata)

		#rate = sampling rate, sig = data; the data will be a two-tuple array format where the first item of each row will be the left channel data, and the second item will be the right channel data
		#This code below is used to extract features from audio samples using MFCC
		newSig = []
		for i in range(len(sig)):newSig.append(sig[i][0])
		newSig = normalize(newSig,200000)
		rate = newSig.shape[0]/sig.shape[0]*rate
		mfcc_feat = mfcc(newSig,rate)#,nfft=)
		d_mfcc_feat = delta(mfcc_feat, 1)
		fbank_feat = logfbank(newSig,rate)
		fbank_feat = fbank_feat.ravel()
		fbank_feat = normalize(fbank_feat,11778) #Normalizing the features extracted
		features.append((normalize(fbank_feat.ravel(),11778),sounddata[:1]))
		Xdir[sounddata.replace(".wav", "")] = len(features) - 1
		X.append(fbank_feat)
		
	y.append(sounddata[:1])
	finish = time()
	print("Time to load data %.3f s" % (finish - start))	
	
	
	#-------------------------------------------------------------------------------------------------------------------------------------------
	#				1. If subject is seen and sentence known but taking some samples out
	#-------------------------------------------------------------------------------------------------------------------------------------------
	#3 accents x 5 subjects x 5 sentences x 10 samples
	#First question: sub known, sent known
	print("Computing Question 1: If subject is seen and sentence known but taking some samples out")
	totalaccents = 4
	subsperaccent = 5
	totalsents = 5
	totalsamples = 10
	trainsamples = 5


	testsamples = totalsamples - trainsamples #Don't change
	#Accent,Subject,Sentence,Sample
	X_train = []
	X_test= []
	Y_train = []
	Y_test = []


	for acc in range(1,totalaccents+1):
		for sub in range(1,subsperaccent+1):
			for sent in range(1,totalsents+1):
				for samp in range(1,totalsamples+1):

					filename = ",".join([str(it) for it in [acc,sub,sent,samp]])
					
					if samp <= trainsamples : 

						X_train.append(X[Xdir[filename]]) #Features of audio sample 1-5
						Y_train.append(int(filename[0])) #labels (1,2,3,4)

					else:			#Otherwise we testing on the remaining

						X_test.append(X[Xdir[filename]]) #Features of audio sample 6-10
						Y_test.append(int(filename[0]))

					
	#feature scaling in order to standardize the features
	scaler = StandardScaler().fit(X_train) 
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)

	#Creating a training file that will contain the training data
	trainfile = "Question1.dat"
	#Cross validation in order to get the best C and Gamma parameter
	custom_dump_svmlight_file(X_train, Y_train, trainfile)
	crange = "-5,13,2"#"1,5,2"
	grange = "-15,5,2"#"-3,2,2"
	
	C,gamma,cvrate = paramsfromexternalgridsearch(trainfile, crange, grange, printlines=True)
	#for 2 accents: best was C=2**3, gamma=2**-15
	clf = SVC(gamma=gamma,C=C, kernel="rbf")
	clf.fit(X_train,Y_train)
	
	#Passing in the test samples against the created model
	modelPrediction = clf.predict(X_test)
	print("The model accuracy is:",metrics.accuracy_score(Y_test,modelPrediction)*100,"%")
	print("precision scores")
	print("Macro: ",precision_score(Y_test, modelPrediction, average='macro')*100,"%")
	print("recall scores")
	print("Macro: ",recall_score(Y_test, modelPrediction, average='macro')*100,"%")
	print("f1 scores")
	print("Macro: ",f1_score(Y_test, modelPrediction, average='macro')*100,"%")
	
	
	writetopcklfile("Question1.model",clf) #Writing the model to a picklefile

	finish = time()
	print("Time to compute Q1 %.3f s" % (finish - start))
	
	#-------------------------------------------------------------------------------------------------------------------------------------------
	#				2.If subject is seen but sentence is unseen
	#-------------------------------------------------------------------------------------------------------------------------------------------
	print("Computing Question 2: If subject is seen but Sentence is unseen")
	totalaccents = 4
	subsperaccent = 5
	totalsents = 5
	train_sentence = 4
	totalsamples = 10

	testsentence= totalsents  - train_sentence #Don't change
	#Accent,Subject,Sentence,Sample
	X_train2 = []
	X_test2 = []
	Y_train2 = []
	Y_test2 = []

	for acc in range(1,totalaccents+1):
		for sub in range(1,subsperaccent+1):
			for sent in range(1,totalsents+1):
				for samp in range(1,totalsamples+1):

					filename = ",".join([str(it) for it in [acc,sub,sent,samp]])

					if sent <= train_sentence: 

						X_train2.append(X[Xdir[filename]]) #Features of audio sample 1-5
						
						Y_train2.append(filename[0]) #labels (1,2,3,4)

					else:			#Otherwise we testing on the remaining

						X_test2.append(X[Xdir[filename]]) #Features of audio sample 6-10
						
						Y_test2.append(filename[0])

	#feature scaling in order to standardize the features				
	scaler = StandardScaler().fit(X_train2)
	X_train2 = scaler.transform(X_train2)
	X_test2 = scaler.transform(X_test2)


	#Creating a training file that will contain the training data
	trainfile = "Question2.dat"

	custom_dump_svmlight_file(X_train2, Y_train2, trainfile)
	
	#Cross validation in order to get the best C and Gamma parameter
	crange = "-5,13,2"	#"1,5,2"
	grange = "-15,5,2"	#"-3,2,2"
	C,gamma,cvrate = paramsfromexternalgridsearch(trainfile, crange, grange, printlines=True)
	clf = SVC(gamma=gamma,C=C, kernel="rbf")
	clf.fit(X_train2,Y_train2)
	
	#Passing in the test samples against the created model
	modelPrediction2 = clf.predict(X_test2)
	print("The model accuracy is:",metrics.accuracy_score(Y_test2,modelPrediction2)*100,"%")
	print("precision scores")
	print("Macro: ",precision_score(Y_test2, modelPrediction2, average='macro')*100,"%")
	print("recall scores")
	print("Macro: ",recall_score(Y_test2, modelPrediction2, average='macro')*100,"%")
	print("f1 scores")
	print("Macro: ",f1_score(Y_test2, modelPrediction2, average='macro')*100,"%")
	
	writetopcklfile("Question2.model",clf) #Writing the model to a picklefile
	
	#-------------------------------------------------------------------------------------------------------------------------------------------
	#				3.If subject is unseen but sentence is seen
	#-------------------------------------------------------------------------------------------------------------------------------------------
	print("Computing Question 3: If subject is unseen but sentence is seen")
	totalaccents = 4
	subsperaccent = 5
	totalsents = 5
	train_subjects = 3
	totalsamples = 10

	testsubject= subsperaccent - train_subjects #Don't change
	#Accent,Subject,Sentence,Sample
	X_train3 = []
	X_test3 = []

	Y_train3 = []
	Y_test3 = []

	for acc in range(1,totalaccents+1):
		for sub in range(1,subsperaccent+1):
			for sent in range(1,totalsents+1):
				for samp in range(1,totalsamples+1):

					filename = ",".join([str(it) for it in [acc,sub,sent,samp]])

					if sent <= train_subjects: 

						X_train3.append(X[Xdir[filename]]) #Features of subjects  1-3
						
						Y_train3.append(filename[0]) #labels (1,2,3,4)

					else:			#Otherwise we testing on the remaining

						X_test3.append(X[Xdir[filename]]) #Features of subjects 3-5
						Y_test3.append(filename[0])


	#feature scaling in order to standardize the features	
	scaler = StandardScaler().fit(X_train3)
	X_train3 = scaler.transform(X_train3)
	X_test3 = scaler.transform(X_test3)
	
	#Creating a training file that will contain the training data
	trainfile = "Question3.dat"

	custom_dump_svmlight_file(X_train3, Y_train3, trainfile)
	#Cross validation in order to get the best C and Gamma parameter
	crange = "-5,13,2"#"1,5,2"
	grange = "-15,5,2"#"-3,2,2"
	C,gamma,cvrate = paramsfromexternalgridsearch(trainfile, crange, grange, printlines=True)
	
	clf = SVC(gamma=gamma,C=C, kernel="rbf")
	clf.fit(X_train3,Y_train3)
	#Passing in the test samples against the created model
	modelPrediction3 = clf.predict(X_test3)
	print("The model accuracy is:",metrics.accuracy_score(Y_test3,modelPrediction3)*100,"%")
	print("precision scores")
	print("Macro: ",precision_score(Y_test3, modelPrediction3, average='macro')*100,"%")
	print("recall scores")
	print("Macro: ",recall_score(Y_test3, modelPrediction3, average='macro')*100,"%")
	print("f1 scores")
	print("Macro: ",f1_score(Y_test3, modelPrediction3, average='macro')*100,"%")
	#writetopcklfile("try3.model",clf) 
	
	
	#-------------------------------------------------------------------------------------------------------------------------------------------
	#				4.If subject is unseen but sentence is unseen
	#-------------------------------------------------------------------------------------------------------------------------------------------
	print("Computing Question 4: subject is unseen but sentence is unseen")
	totalaccents = 4
	subsperaccent = 5
	totalsents = 5
	train_sentence = 4
	train_subjects = 4
	totalsamples = 10

	testsubject= subsperaccent - train_subjects
	test_sentence = totalsents  - train_sentence	#Don't change
	#Accent,Subject,Sentence,Sample
	X_train4 = []
	X_test4 = []
	Y_train4 = []
	Y_test4 = []


	for acc in range(1,totalaccents+1):
		for sub in range(1,subsperaccent+1):
			for sent in range(1,totalsents+1):
				for samp in range(1,totalsamples+1):

					filename = ",".join([str(it) for it in [acc,sub,sent,samp]])

					if sub <= train_subjects and sent <= train_sentence: 

						X_train4.append(X[Xdir[filename]]) 
						#X_train_labels3.append(filename)
						Y_train4.append(filename[0]) #labels (1,2,3)

					else:			#Otherwise we testing on the remaining

						X_test4.append(X[Xdir[filename]]) #Features of subjects 3-5
						#X_test_labels3.append(filename)  #contains subjects 3-5
						Y_test4.append(filename[0])

	
	#feature scaling in order to standardize the features	
	scaler = StandardScaler().fit(X_train4)
	X_train4 = scaler.transform(X_train4)
	X_test4 = scaler.transform(X_test4)

	trainfile = "Question4.dat"

	custom_dump_svmlight_file(X_train4, Y_train4, trainfile)
	
	#Cross validation in order to get the best C and Gamma parameter
	crange = "-5,13,2" #"1,5,2"
	grange = "-15,5,2" #"-3,2,2"
	C,gamma,cvrate = paramsfromexternalgridsearch(trainfile, crange, grange, printlines=True)
	#for 2 accents: best was C=2**3, gamma=2**-15
	
	clf = SVC(gamma=gamma,C=C, kernel="rbf")
	clf.fit(X_train4,Y_train4)
	
	#Passing in the test samples against the created model
	modelPrediction4 = clf.predict(X_test4)
	
	print("The model accuracy is:",metrics.accuracy_score(Y_test4,modelPrediction4)*100,"%")
	print("precision scores")
	print("Macro: ",precision_score(Y_test4, modelPrediction4, average='macro')*100,"%")
	print("recall scores")
	print("Macro: ",recall_score(Y_test4, modelPrediction4, average='macro')*100,"%")
	print("f1 scores")
	print("Macro: ",f1_score(Y_test4, modelPrediction4, average='macro')*100,"%")
	
	writetopcklfile("Question4.model",clf)
	
main()



