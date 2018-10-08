import os

NUMBER_OF_WORDS = 24748
EPS = 0.01
threshold = 16.1
os.chdir(path='./pu1')
folds = os.listdir(path='.')

N = len(folds)


def toInt(lst):
    ans = []
    for str in lst:
        if(str.isdigit()):
            ans.append(int(str))
    return ans

def isSpam(filename):
    return (filename.find('spmsg') != -1)

glob_error1 = 0
glob_error2 = 0
glob_correct = 0
final = 0
correct = 0
error1 = 0
error2 = 0

for test_fold in folds:
    used_subj = [[EPS, EPS] for i in range(NUMBER_OF_WORDS)]
    used_body = [[EPS, EPS] for i in range(NUMBER_OF_WORDS)]
    messages = [0, 0]
    for fold in folds:
        if(fold != test_fold):
            os.chdir(path = './' + fold)
            filenames = os.listdir(path='.')
            for file in filenames:
                f = open(file, "r")
                strn = f.read()
                strn = strn.split('\n')
                subject = strn[0].split(' ')
                subject = toInt(subject)
                body = strn[2].split(' ')
                body = toInt(body)
                spflag = isSpam(file)
                messages[spflag] = messages[spflag] + 1
                for k in range(len(subject)):
                    used_subj[subject[k]][spflag] = used_subj[subject[k]][spflag] + 1
                for k in range(len(body)):
                    used_body[body[k]][spflag] = used_body[body[k]][spflag] + 1
                f.close()
            os.chdir(path = '..')
    print(messages)
    spam_prob = float(messages[1] / (sum(messages)))
    os.chdir(path = './' + test_fold)
    filenames = os.listdir(path='.')
    for file in filenames:
        f = open(file, "r")
        strn = f.read()
        strn = strn.split('\n')
        subject = strn[0].split(' ')
        subject = toInt(subject)
        body = strn[2].split(' ')
        body = toInt(body)
        spflag = isSpam(file)
        prob_subj = []
        prob_body = []
        p1, p2 = 0, 0
        for word in subject:
            pxy = used_subj[word][1] / (used_subj[word][1] + used_subj[word][0])
            p1 = p1 + spam_prob * pxy
            p2 = p2 + (1 - spam_prob) * (1 - pxy)
        for word in body:
            pxy = used_body[word][1] / (used_body[word][1] + used_body[word][0])
            p1 = p1 + spam_prob * pxy
            p2 = p2 + (1 - spam_prob) * (1 - pxy)
        if (p1 - p2 > threshold):
            if(spflag):
                correct = correct + 1
            else:
                print(p1 - p2)
                error1 = error1 + 1
        else:
            if (spflag):
                error2 = error2 + 1
            else:
                correct = correct + 1
    os.chdir(path = '..')

print(str(error1) + ' normal messages classified as spam')
print(str(error2) + ' spam messages classified as normal')
print(str(float(correct/(correct + error1 + error2))) + '% of correct answers')


