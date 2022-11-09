from numpy import genfromtxt
import os
import pandas as pd
import random
# read from the hdf5 file
import h5py
import csv


emos = ['col', 'dis', 'gio', 'pau', 'rab', 'sor', 'tri']
counts = [0] * 7
emo_div = {}
emo_gender = {}
divisions = ['train', 'dev', 'test']
for d in divisions:
    emo_div[d] = dict(zip(emos, counts))
genders = ['f', 'm']
for g in genders:
    emo_gender[g] = dict(zip(emos, counts))


clients = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69']
fs = set()
ms = set()

for index in range(0, 9365):
    with h5py.File('./logmel_demos.hdf5', 'r') as hf:
        audio_name = hf['audio_name'][index].decode()
        gender = hf['gender'][index].decode()
        idnumber = hf['speaker_id'][index].decode()

        if gender == 'f':
            fs.add(idnumber)
        else:
            ms.add(idnumber)

    if len(fs) == 23 and len(ms) == 45:
        break

random.seed(123)
# split of the female clients, 40%, 30%, 30% for train, dev, and test
train_f = random.sample(list(fs), 9)
dev_f = random.sample(list(fs - set(train_f)), 7)
test_f = list(fs - set(train_f) - set(dev_f))

# split of the male clients, 40%, 30%, 30% for train, dev, and test
train_m = random.sample(list(ms), 18)
dev_m = random.sample(list(ms - set(train_m)), 18)
test_m = list(ms - set(train_m) - set(dev_m))

train = []
dev = []
test = []
for index in range(0, 9365):
    with h5py.File('./logmel_demos.hdf5', 'r') as hf:
        audio_name = hf['audio_name'][index].decode()
        gender = hf['gender'][index].decode()
        idnumber = hf['speaker_id'][index].decode()
        emotion = hf['emotion'][index].decode()

        if idnumber in train_f or idnumber in train_m:
            train.append(audio_name)
            emo_div['train'][emotion] += 1
        elif idnumber in dev_f or idnumber in dev_m:
            dev.append(audio_name)
            emo_div['dev'][emotion] += 1
        else:
            test.append(audio_name)
            emo_div['test'][emotion] += 1

        # Gender-wise emotion distribution
        if gender == 'f':
            emo_gender['f'][emotion] += 1
        elif gender == 'm':
            emo_gender['m'][emotion] += 1

df = pd.DataFrame(train)
df.to_csv("split_csv/demos/train.csv", index=False, header=None)

df = pd.DataFrame(dev)
df.to_csv("split_csv/demos/dev.csv", index=False, header=None)

df = pd.DataFrame(test)
df.to_csv("split_csv/demos/test.csv", index=False, header=None)

df = pd.DataFrame.from_dict(emo_div)
df.to_csv("split_csv/demos/emo_div.csv", index=False, header=None)

df = pd.DataFrame.from_dict(emo_gender)
df.to_csv("split_csv/demos/emo_gender.csv", index=False, header=None)

'''
note1 = 'Clients gender female: ' + ', '.join(list(fs))
note2 = 'Clients gender male: ' + ', '.join(list(ms))
note3 = 'Clients in pre-train: ' + ', '.join(pre_f) + ', ' + ', '.join(pre_m)
note4 = 'Clients in train: ' + ', '.join(train_f) + ', ' + ', '.join(train_m)
note5 = 'Clients in test: ' + ', '.join(test_f) + ', ' + ', '.join(test_m)
'''
note1 = 'Number of samples in train is {}, dev is {}, test is {}'.format(len(train), len(dev), len(test))
with open('split_csv/demos/div.txt','w') as f:
    f.write('{}\n'.format(note1))
'''
with open('split_csv/demos/note.txt','w') as f:
    f.write('{}\n{}\n{}\n{}\n{}\n{}\n'.format(note1,note2,note3,note4,note5,note6))
''
