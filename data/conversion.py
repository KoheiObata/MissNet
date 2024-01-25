import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os

seed = 0
random.seed(seed)
np.random.seed(seed)


def make_dir(input_dir):
    if os.path.isdir(input_dir):
        print(f'{input_dir} already exist')
    else:
        os.makedirs(f"{input_dir}")
        print(f'{input_dir} is ready')

def plot_indicator(W, savepath=False):
    #missing values are black
    sns.heatmap(W.T, cbar=False)
    if savepath:
        sns_plot = sns.heatmap(W.T, cbar=False)
        fig = sns_plot.get_figure()
        fig.savefig(savepath)
        plt.close()

def missing_matrix_rate(X, max_missing_len=0.20, min_missing_len=0.0, missing_rate=0.1):
    #generate missing blocks
    #X = (Time, dim)
    #if max/min_missing_len < 1: rate， elif >1: length
    #max_missing_rate: rate of missing values per the whole length (Time)

    #W: if X[i,j] == nan then W[i,j] == 0 else W[i,j] == 1
    T,D = X.shape
    max_len = max_missing_len if max_missing_len>=1 else int(T*max_missing_len)
    min_len = min_missing_len if min_missing_len>=1 else int(T*min_missing_len)
    print('max missing block length', max_len)
    print('min missing block length', min_len)

    W = ~np.isnan(X)
    n_nan = 0
    count = 0
    print('X.size', X.size)
    print('missing_rate', missing_rate)
    print('target missing size', X.size*missing_rate)
    print('initial missing', n_nan)
    while np.count_nonzero(W==0)-n_nan<X.size*missing_rate:
        count += 1
        if count > 100000:
            print('Maybe it ends up in an endless loop')
            break
        # missing_point[row, startpoint, length]
        missing_point = [random.randint(0,D-1), random.randint(-max_len,T+max_len), random.randint(min_len,max_len)]
        start = missing_point[1]
        end = missing_point[1] + missing_point[2]
        if end>=T:
            end=T-1
        elif end<0:
            end=0
        if start>=T:
            start=T-1
        elif start<0:
            start=0
        #avoid a blackout sequence
        W_copy = np.copy(W)
        W_copy[start:end, missing_point[0]] = 0
        if np.all(W_copy[:, missing_point[0]]==0):
            continue
        else:
            W = W_copy
    print('total number of missing',np.count_nonzero(W==0))
    return W


def convert_List(site, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train):
    if site=='dynammo':
        name_list = ['127_07',  '132_32',  '132_43',  '132_46',  '135_07',  '141_04',  '141_16',  '143_22',  '80_10']
        head = f'{site}'
    elif site=='motes':
        name_list=['motes']
        head = ''
    elif site=='synthetic/pattern':
        name_list = ['10_50_1000_1', '10_50_1000_2']
        new_name_list = []
        num = 5
        for name in name_list:
            for i in range(num):
                new_name = name + f'/num={i}'
                new_name_list.append(new_name)
        name_list = new_name_list
        head = f'{site}'
    elif site=='synthetic/scale_len':
        name_list = []
        for len in [100,250,500,1000,2500,5000,10000,25000,50000,100000,250000,500000]:
            name_list.append(f'10_50_{len}_1')
        head = f'{site}'
    elif site=='synthetic/scale_dim':
        name_list = []
        for d1 in range(20,401,20):
            name_list.append(f'10_{d1}_1000_1')
        head = f'{site}'

    for name in name_list:
        name = f'{head}/{name}'
        X_true, W_true, W_test, W_train = convert(name, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train)
    return X_true, W_true, W_test, W_train

def convert(name, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train):
    filename = name.split('/')[-1]
    PATH = f'./original/{name}'
    SAVEPATH = f'./experiment/{name}/MML={max_missing_len}_{min_missing_len}/MRTe={missing_rate_test}/MRTr={missing_rate_train}'
    if not os.path.isfile(f'{PATH}/data.txt'):
        print(PATH, 'has no file')
        return 0, 0, 0, 0

    if os.path.isfile(f'{SAVEPATH}/X_true.txt'):
        print(SAVEPATH, 'exists')
        return 0, 0, 0, 0
    make_dir(SAVEPATH)
    X = load_data(PATH, filename)
    X_true, W_true, W_test, W_train = generate_missing(X, SAVEPATH, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train)
    return X_true, W_true, W_test, W_train

def load_data(PATH, filename):
    try:
        X = np.loadtxt(f'{PATH}/data.txt')
    except:
        X = pd.read_csv(f'{PATH}/data.txt', delimiter=',', index_col=0)
        if 'motes' in filename or 'Motes' in filename:
            X = X.drop(['5'], axis=1)
        X = X.to_numpy()
    return X

def generate_missing(X, SAVEPATH, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train):

    W_true = ~np.isnan(X)
    X_true = np.copy(X)
    X_true[W_true == 0] = np.nan
    np.savetxt(f'{SAVEPATH}/W_true.txt', W_true, fmt='%i')
    plot_indicator(W_true, savepath=f'{SAVEPATH}/W_true.png')

    print('Create missing test')
    W_test = missing_matrix_rate(X, max_missing_len=max_missing_len, min_missing_len=min_missing_len, missing_rate=missing_rate_test)
    X_test = np.copy(X)
    X_test[W_test == 0] = np.nan
    np.savetxt(f'{SAVEPATH}/W_test.txt', W_test, fmt='%i')
    plot_indicator(W_test, savepath=f'{SAVEPATH}/W_test.png')

    print('Create missing train')
    if missing_rate_train>0:
        test_nan_rate = np.count_nonzero(W_test==0)/W_test.size
        W_train = missing_matrix_rate(X_test, max_missing_len=max_missing_len, min_missing_len=min_missing_len, missing_rate=test_nan_rate+missing_rate_train)
    else:
        W_train = W_test
    X_train = np.copy(X)
    X_train[W_train == 0] = np.nan
    np.savetxt(f'{SAVEPATH}/W_train.txt', W_train, fmt='%i')
    plot_indicator(W_train, savepath=f'{SAVEPATH}/W_train.png')

    #Z-score normalization
    mean = np.nanmean(X_true,axis=0)
    std = np.nanstd(X_true,axis=0)
    X_true = (X_true-mean)/std
    X_test = (X_test-mean)/std
    X_train = (X_train-mean)/std

    np.savetxt(f'{SAVEPATH}/X_true.txt', X_true)
    np.savetxt(f'{SAVEPATH}/X_test.txt', X_test)
    np.savetxt(f'{SAVEPATH}/X_train.txt', X_train)

    return X_true, W_true, W_test, W_train




max_missing_len = 0.05
min_missing_len = 0.0
missing_rate_train = 0.1


# for missing_rate_test in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
for missing_rate_test in [0.2]:
    site = 'dynammo'
    X_true, W_true, W_test, W_train = convert_List(site, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train)

    name='motes'
    X_true, W_true, W_test, W_train = convert(name, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train)



missing_rate_train = 0.0
# for missing_rate_test in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]:
for missing_rate_test in [0.2]:
    site = 'synthetic/pattern'
    X_true, W_true, W_test, W_train = convert_List(site, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train)

if False:
    for missing_rate_test in [0.4]:
        site = 'synthetic/scale_len'
        X_true, W_true, W_test, W_train = convert_List(site, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train)
        site = 'synthetic/scale_dim'
        X_true, W_true, W_test, W_train = convert_List(site, max_missing_len, min_missing_len, missing_rate_test, missing_rate_train)