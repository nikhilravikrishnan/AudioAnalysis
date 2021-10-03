
import sys
sys.path.append('/home/nikhil/Prosody/pyAudioAnalysis/pyAudioAnalysis/')
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioTrainTest as aT
from pyAudioAnalysis import MidTermFeatures as aF
from pyAudioAnalysis import audioAnalysis as aA
from pyAudioAnalysis import ShortTermFeatures as sF
from pathlib import Path
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from glob import glob
import os



def extract_features_and_train(features, classifier_type, model_name, train_percentage=0.90):

    n_feats = features[0].shape[1]
    feature_names = ["features" + str(d + 1) for d in range(n_feats)]

    # STEP B: classifier Evaluation and Parameter Selection:
    if classifier_type == "svm" or classifier_type == "svm_rbf":
        classifier_par = np.array([0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0])
    elif classifier_type == "randomforest":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])
    elif classifier_type == "knn":
        classifier_par = np.array([1, 3, 5, 7, 9, 11, 13, 15])        
    elif classifier_type == "gradientboosting":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])
    elif classifier_type == "extratrees":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])

    # get optimal classifeir parameter:
    temp_features = []
    for feat in features:
        temp = []
        for i in range(feat.shape[0]):
            temp_fv = feat[i, :]
            if (not np.isnan(temp_fv).any()) and (not np.isinf(temp_fv).any()):
                temp.append(temp_fv.tolist())
            else:
                print("NaN Found! Feature vector not used for training")
        temp_features.append(np.array(temp))
    features = temp_features
#     print(f'##########'+classifier_par+'############')
    print(f'Classifier Parameters:{classifier_par.shape}')
    best_param, confusion_matrix, best_score = aT.evaluate_classifier(features, class_names, 100, classifier_type, classifier_par, 1, train_percentage)

    print("Selected params: {0:.5f}".format(best_param))

    return best_score



if (__name__=='__main__'):
    directory_in_str = "/home/nikhil/Prosody/GuitarDataste/"
    pathlist = Path(directory_in_str).glob('**/*.wav')

    listOfDirs = []
    for path in pathlist:
        path_in_str = str(path)
        parent = path.parents[0]
        if str(parent) not in listOfDirs:
            listOfDirs.append(str(parent))

    mtWin = 1.0
    mtStep = 1.0
    stWin = aT.shortTermWindow
    stStep = aT.shortTermStep
    classifierType = "svm"
    modelName = "models/svm_1v1_1.0.0"
    beat = False


    all_files = []
    for folder in listOfDirs:
        files = glob(folder+'/*')
        for file in files:
            all_files.append(file)
    #         print(file)
    #         cmd = f'ffmpeg -i "{file}" -ar 16000 "{file}" -y'
    #         print(cmd)
    #         os.system(cmd)


    all_classes = []
    for path in all_files:
    #     print(path)
        all_classes.append(path.split(os.sep)[-1][:-7])

    all_classes = np.unique(all_classes)

    classes = {}
    for emotion in all_classes:
        classes[emotion] = []
    for file in all_files:
        for emotion in all_classes:
            if emotion in file:
                classes[emotion].append(file)


    unwanted = ['Angry:Anxi','Angry_Anxious#0', 'Angry_Anxious#1','Compassion Com', 'Compassion Comp ',
            'Disappointm','Fear Tak', 'Inter','Jo','Pr','Regret Com', 'Regret Comp ', 'Sh', '', 'Amusem', 'Amusemen', 'Happy_Excited', 'Relaxed_Content',
                'Angry_Anxious', 'Sad_Bored', 'Very sh', 'Contempt', 'Contentment', 'Disgust',  'Pride', 'Regret', 'Relief', 'Shame', 'Amusement', 'Hate', 'Pleasure']

    for item in unwanted:
        del classes[item]

    print(classes.keys())
    features, class_names, file_names, feature_names = aF.multiple_directory_feature_extraction(classes, mtWin, mtStep, stWin, stStep, beat)

    features_copy = [features[i] for i in range(len(features))]
    feature_names_copy = feature_names

    perm_features = []
    accuracies = []
    perm_feature_names = []

    print(features_copy[0].shape)
    print(feature_names_copy)


    while feature_names_copy:
        temp_accs = []
        for i in range(len(feature_names_copy)):
            print("\nEvaluating {} | {} / {}".format(feature_names_copy[i], i, len(feature_names_copy)))
            if perm_features:
                train_features = [np.concatenate((perm_features[j], features_copy[j][:,i,np.newaxis]), axis=1) for j in range(len(features_copy))]
            else:
                train_features = [features_copy[j][:,i,np.newaxis] for j in range(len(features_copy))]
            print(train_features[0].shape)
            temp_accs.append(extract_features_and_train(train_features, "svm", "models/tempSvm_1v1_1.0.0"))        
        best_acc = max(temp_accs)
        best_index = temp_accs.index(best_acc)
        accuracies.append(best_acc)
        if perm_features:
            perm_features = [np.concatenate((perm_features[j], features_copy[j][:,best_index,np.newaxis]), axis=1) for j in range(len(features_copy))]
        else:
            perm_features = [features_copy[j][:,best_index,np.newaxis] for j in range(len(features_copy))]
        features_copy = [np.delete(features[i], best_index, axis=1) for i in range(len(features_copy))]
        added_feature = feature_names_copy.pop(best_index)
        perm_feature_names.append(added_feature)
        print("Adding {} to the perm features".format(added_feature))
        print("New accuracy is {}".format(best_acc))
        print("New Perm Features list is {}".format(perm_feature_names))
