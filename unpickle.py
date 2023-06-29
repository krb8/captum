import os
import pickle as cPickle
import numpy
example_dict = {'num_cases_per_batch': 10, 'label_names': ["stop sign", "yield", "speed limit", "cat", "deer", "dog", "frog", "horse", "ship", "truck"], 'num_vis': 3072}
#file = r"/home/krb8/NISTworkTest/captum/captum/insights/attr_vis/data/test/cifar-10-batches-py/batches.meta"

## pass in uniqueLabelList (this includes classes from ALL BATCHES)
def allLabelNames(uniqueLabelList):
    print("UNIQUE LABEL LIST: ", uniqueLabelList)
    ordered = sorted(uniqueLabelList)
    print("ordered unique labels: ", ordered)
    label_names = []
    for e in ordered:
        label_names.append("TrojAI Class {} ".format(e))
    print(label_names)













f=r'C:\Users\jmess\OneDrive\Desktop\summerWork\outputTest\data_batch_1.npy'
def unpickle(file):
    #import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

#entry = pickle.load(f, encoding="latin1")
def unNumpy(f):
    with open(f, 'rb') as fo:
        dict = numpy.load(fo, allow_pickle=True)
    return dict

#print(unNumpy(f))

def Pickle(fileName):
    print("pickling: ", fileName)
    pickle_out = open(fileName,"wb")
    cPickle.dump(example_dict, pickle_out)
    pickle_out.close()

outF = r"C:\Users\jmess\OneDrive\Desktop\summerWork\data2\test2\cifar-10-batches-py\batches.meta"
inF=r'C:\Users\jmess\OneDrive\Desktop\summerWork\captum\batches.meta'
def saveToNumpy(inFilePath):
    inFileContent = None

    outfp = os.path.join(r'C:\Users\jmess\OneDrive\Desktop\summerWork\captum\batches.meta')
    with open(inFilePath,'r') as file:
        inFileContent = file.read()

    print("file content input ", inFileContent)
    numpy.save(outfp, inFileContent, allow_pickle=True)

##gives a batches.meta.npy (clean later) and save to right directory

#saveToNumpy(inF)


thing = {'label_names': ['TrojAI Class 0 ', 'TrojAI Class 1 ', 'TrojAI Class 2 ', 'TrojAI Class 3 ', 'TrojAI Class 4 ', 'TrojAI Class 5 ', 'TrojAI Class 6 ', 'TrojAI Class 7 ', 'TrojAI Class 8 ', 'TrojAI Class 9 ', 'TrojAI Class 10 ', 'TrojAI Class 11 ', 'TrojAI Class 12 ', 'TrojAI Class 13 ', 'TrojAI Class 14 ', 'TrojAI Class 15 ', 'TrojAI Class 16 ', 'TrojAI Class 17 ', 'TrojAI Class 18 ', 'TrojAI Class 19 ', 'TrojAI Class 20 ', 'TrojAI Class 21 ', 'TrojAI Class 22 ', 'TrojAI Class 23 ', 'TrojAI Class 24 ', 'TrojAI Class 25 ', 'TrojAI Class 26 ', 'TrojAI Class 27 ', 'TrojAI Class 28 ', 'TrojAI Class 29 ', 'TrojAI Class 30 ', 'TrojAI Class 31 ', 'TrojAI Class 32 ', 'TrojAI Class 33 ', 'TrojAI Class 34 ', 'TrojAI Class 35 ', 'TrojAI Class 36 ', 'TrojAI Class 37 ', 'TrojAI Class 38 ', 'TrojAI Class 39 ', 'TrojAI Class 40 ', 'TrojAI Class 41 ', 'TrojAI Class 42 ', 'TrojAI Class 43 ', 'TrojAI Class 44 ', 'TrojAI Class 45 ', 'TrojAI Class 46 ', 'TrojAI Class 47 ', 'TrojAI Class 48 ', 'TrojAI Class 49 ', 'TrojAI Class 50 ', 'TrojAI Class 51 ', 'TrojAI Class 52 ', 'TrojAI Class 53 ', 'TrojAI Class 54 ', 'TrojAI Class 55 ', 'TrojAI Class 56 ', 'TrojAI Class 57 ', 'TrojAI Class 58 ', 'TrojAI Class 59 ', 'TrojAI Class 60 ', 'TrojAI Class 61 ', 'TrojAI Class 62 ', 'TrojAI Class 63 ', 'TrojAI Class 64 ', 'TrojAI Class 65 ', 'TrojAI Class 66 ', 'TrojAI Class 67 ', 'TrojAI Class 68 ', 'TrojAI Class 69 ', 'TrojAI Class 70 ', 'TrojAI Class 71 ', 'TrojAI Class 72 ', 'TrojAI Class 73 ', 'TrojAI Class 74 ', 'TrojAI Class 75 ', 'TrojAI Class 76 ', 'TrojAI Class 77 ', 'TrojAI Class 78 ', 'TrojAI Class 79 ', 'TrojAI Class 80 ', 'TrojAI Class 81 ', 'TrojAI Class 82 ', 'TrojAI Class 83 ', 'TrojAI Class 84 ', 'TrojAI Class 85 ', 'TrojAI Class 86 ', 'TrojAI Class 87 ', 'TrojAI Class 88 ', 'TrojAI Class 89 ', 'TrojAI Class 90 ', 'TrojAI Class 91 ', 'TrojAI Class 92 ', 'TrojAI Class 93 ', 'TrojAI Class 94 '] }

# def saveDictToNumpy(dict):
#     outfp = os.path.join(r'C:\Users\jmess\OneDrive\Desktop\summerWork\captum\batches.meta')
#     print("file content input ", dict)
#     numpy.save(outfp, dict, allow_pickle=True)

#saveDictToNumpy(thing)
