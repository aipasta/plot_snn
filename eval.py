import os
import torch
import numpy as np 
import math
import matplotlib.pyplot as plt


def load_hprof_from_files(path_dir=''):
    file_list = os.listdir(path_dir)
    hprof_list = list()

    for f in file_list:
        filename = os.path.splitext(f)  # split filename and ext

        if not filename[1] == '.pth':  # read only pth files
            continue

        hp_dict = dict()
        for v in filename[0].split('_'):
            hp_dict['filename']=os.path.join(path_dir, f)
            k = v.split('-')
            hp_dict[k[0]] = k[1]
        hprof_list.append(hp_dict)
    return hprof_list

def compare_hprof(x, y):
    x_keys = set(x.keys())
    y_keys = set(y.keys())
    intersect_keys = x_keys.intersection(y_keys)
    keys_diff = x_keys.union(y_keys) - intersect_keys

    if len(keys_diff - {'filename','niter', 'time'}) > 0:
        return False

    for k in intersect_keys-{'niter', 'time'}:
        if not x[k] == y[k]:
            return False
    
    return True

def get_hprof_separate_iter(hp_list, input_hp, time_range=None):
    ret_dict = dict()
    for hp in hp_list:
        if compare_hprof(hp, input_hp):
            if not hp['time'] in ret_dict.keys():
                ret_dict[hp['time']] = list()
            ret_dict[hp['time']].append(hp)

    # print(ret_dict)
    if len(ret_dict) == 0:
        print('There is no matched file', input_hp)
        return None

    if time_range == None:  # return latest ones
        latest_key = sorted(ret_dict.keys())[-1]
        return ret_dict[latest_key]
    else:
        for t in reversed(sorted(ret_dict.keys())):
            if time_range[0] <= t <= time_range[1]:
                return ret_dict[t]

    print('There is no matched file', input_hp, time_range)
    return None

def get_stat_for_hprof(hprofs, metrics=None):
    if hprofs == None:
        return -1
    ret = dict()
    eval_data = dict()
    if metrics == None:
        metrics = ['loss_output_train_t', 'acc_train_t']

    for i, h in enumerate(hprofs):
        load_result = torch.load(h['filename'])
        for m in metrics:
            # d = load_result[m]  # TODO
            d = load_result[m][0][0][0][0][0][0][0][0].numpy()  # TODO: temp

            if i == 0:
                len_timestep = 10 # TODO: Get this 
                eval_data[m] =  np.empty((len_timestep,0))
            
            eval_data[m] = np.concatenate((eval_data[m], d), axis=-1)
    
    # print(eval_data) # KDW
    for m in metrics:
        ret[m] = dict()
        ret[m]['num_of_iter'] = len(hprofs)
        ret[m]['mertic'] = m
        ret[m]['timestep'] = np.arange(10) # TODO
        ret[m]['mean'] = np.mean(eval_data[m], axis=-1)
        ret[m]['var'] = np.var(eval_data[m], axis=-1)
        ret[m]['std'] = np.std(eval_data[m], axis=-1)
        z = 1.96  # it is for 95%
        ret[m]['confidence_interval'] = z * np.sqrt(ret[m]['var']) / np.sqrt(ret[m]['num_of_iter'])
    # print(ret)

    return ret

def plot_one_feature(r_list, metrics, label=None):

    if not len(r_list) == len(label):
        print('Number of label is not matched to data')
        label = None
    num_metric = len(metrics)
    fig_width = 3.6 * num_metric
    fig_height = 2.4 * 2
    fig, axs = plt.subplots(2, num_metric, figsize=[fig_width, fig_height])

    for i, m in enumerate(metrics):
        for j, r in enumerate(r_list):
            x = r[m]['timestep']
            y = r[m]['mean']
            # plt.plot(x, y)
            if label == None:
                axs[0][i].plot(x, y) 
            else:
                axs[0][i].plot(x, y, label=label[j])
                axs[0][i].legend()
            
            lb = r[m]['mean'] - r[m]['confidence_interval']
            ub = r[m]['mean'] + r[m]['confidence_interval']
            axs[0][i].fill_between(x, lb, ub, alpha=0.4)

            axs[1][i].bar(j, r[m]['mean'][-1])

        axs[0][i].title.set_text(m)
    plt.show()

def get_result(hp_list, sweep, metrics, input_hp, time_range=None):

    results = list()
    label = list()
    for s in sweep['value']:
        input_hp[sweep['hyperparam']] = str(s)

        target_hprofs = get_hprof_separate_iter(hp_list, input_hp, time_range)
        if target_hprofs == None:
            continue
        results.append(get_stat_for_hprof(target_hprofs, metrics))
        label.append(sweep['hyperparam']+'_'+str(s))

    # print(results)
    plot_one_feature(results, metrics, label)

    return 0

def get_result2(hp_list, sweep1, sweep2, metrics, input_hp, time_range=None):

    result_dict = dict()
    label = list()
    for s1 in sweep1['value']:
        input_hp[sweep1['hyperparam']] = str(s1)
        result_dict[s1] = dict()
        for s2 in sweep2['value']:
            input_hp[sweep2['hyperparam']] = str(s2)

            target_hprofs = get_hprof_separate_iter(hp_list, input_hp, time_range)
            if target_hprofs == None:
                continue
            result_dict[s1][s2] = get_stat_for_hprof(target_hprofs, metrics)
            # results.append(get_stat_for_hprof(target_hprofs, metrics))
            # label.append(sweep['hyperparam']+'_'+str(s))

    # print(results)
    plot_one_feature2(result_dict, metrics)

    return 0

def plot_one_feature2(r_dict, metrics, label=None):

    num_metric = len(metrics)
    fig_width = 3.6 * num_metric
    fig_height = 2.4 * 2
    fig, axs = plt.subplots(2, num_metric, figsize=[fig_width, fig_height])

    for i, m in enumerate(metrics):
        for k1, v in r_dict.items():
            for k2, r in v.items():
                x = r[m]['timestep']
                y = r[m]['mean']
                # plt.plot(x, y)
                if label == None:
                    axs[0][i].plot(x, y) 
                else:
                    axs[0][i].plot(x, y, label=label[j])
                    axs[0][i].legend()
                
                lb = r[m]['mean'] - r[m]['confidence_interval']
                ub = r[m]['mean'] + r[m]['confidence_interval']
                axs[0][i].fill_between(x, lb, ub, alpha=0.4)

                axs[1][i].bar(j, r[m]['mean'][-1])

        axs[0][i].title.set_text(m)
    plt.show()

if __name__ == "__main__":

    input_hprof_1 = { 
    'task': 'prediction',
    'digits': '2',
    'ex': 'single',
    'dec': 'majority',
    'eptrain': '10',
    'eptest': '30',
    'mode': 'iw',
    'niter': '3',
    'Nh': '2',
    'Nk': '10',
    'lr': '0.05',
    'lrconst': '1.5',
    'kappa': '0.05',
    'Nb': '8',
    'nll': '10',
    'time': '03131600'
    }    
    input_hprof_2 = { 
    'task': 'prediction',
    'digits': '2',
    'ex': 'single',
    'dec': 'majority',
    'eptrain': '30',
    'eptest': '30',
    'mode': 'iw',
    'niter': '3',
    'Nh': '2',
    'Nk': '10',
    'lr': '0.05',
    'lrconst': '1.5',
    'kappa': '0.05',
    'Nb': '8',
    'nll': '10'
    }    

    metrics = ['loss_output_train_t', 'acc_train_t']
    time_range = ['03131200', '03131830']
        
    hprof_file_list = load_hprof_from_files('res')

    # target_hprofs_1 = get_hprof_separate_iter(hprof_file_list, input_hprof_1, time_range)
    # r_1 = get_stat_for_hprof(target_hprofs_1, metrics)

    # target_hprofs_2 = get_hprof_separate_iter(hprof_file_list, input_hprof_2)
    # r_2 = get_stat_for_hprof(target_hprofs_2, metrics)

    # plot_one_feature([r_1, r_2], metrics)

    sweep = {'hyperparam':'eptrain', 'value': [10, 30, 40]}
    get_result(hprof_file_list, sweep, metrics, input_hprof_1, time_range)
    
    # sweep1 = {'hyperparam':'eptrain', 'value': [10, 30, 40]}
    # sweep2 = {'hyperparam':'eptest', 'value': [10, 30, 40]}
    # get_result_2(hprof_file_list, sweep1, sweep2, metrics, input_hprof_1, time_range)
