# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pyformat: disable

import os
import scipy.stats

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import functools

# Look at me being proactive!
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def sweep(score, x):
    """
    Compute a ROC curve and then return the FPR, TPR, AUC, and ACC.
    """
    '''
    没啥特别的，就硬算
    '''
    fpr, tpr, _ = roc_curve(x, -score)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    return fpr, tpr, auc(fpr, tpr), acc


def load_data(p):
    """
    Load our saved scores and then put them into a big matrix.
    """
    global scores, keep
    scores = []
    keep = []

    for root, ds, _ in os.walk(p):
        for f in ds:
            if not f.startswith("experiment"): continue
            if not os.path.exists(os.path.join(root, f, "scores")): continue
            print(print(ds))
            last_epoch = sorted(os.listdir(os.path.join(root, f, "scores")))
            if len(last_epoch) == 0: continue
            scores.append(np.load(os.path.join(root, f, "scores", last_epoch[-1])))
            keep.append(np.load(os.path.join(root, f, "keep.npy")))

    scores = np.array(scores)
    print(scores)
    keep = np.array(keep)[:, :scores.shape[1]]

    return scores, keep


def generate_ours(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,
                  fix_variance=False):
    """
    Fit a two predictive models using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    '''
    Fits two predictive models using the given scores and keep parameters to predict if the examples were training 
    data or not, using the ground truth answer from the check_keep parameter.
    '''
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        # a. Append the scores of the examples that are part of the training set (i.e., keep[:, j] is True) to dat_in.
        # b. Append the scores of the examples that are not part of the training set (i.e., keep[:, j] is False)
        #    to dat_out.
        # 注意此时每次j可能都不一样，list dim 不一样
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    in_size = min(min(map(len, dat_in)), in_size)
    out_size = min(min(map(len, dat_out)), out_size)

    dat_in = np.array([x[:in_size] for x in dat_in])
    dat_out = np.array([x[:out_size] for x in dat_out])
    '''
    上面在干什么？
    ans:
    # Example dat_in and dat_out lists
    dat_in = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11, 12],
        [13, 14, 15, 16]
    ]
    
    dat_out = [
        [17, 18, 19, 20, 21],
        [22, 23, 24, 25, 26, 27],
        [28, 29, 30, 31, 32, 33, 34]
    ]
    
    # Determine the minimum number of training examples (in_size) and non-training examples (out_size) across all models
    in_size = min([len(x) for x in dat_in])
    out_size = min([len(x) for x in dat_out])
    
    # Truncate the dat_in and dat_out lists accordingly
    truncated_dat_in = [x[:in_size] for x in dat_in]
    truncated_dat_out = [x[:out_size] for x in dat_out]
    
    print("Truncated dat_in:", truncated_dat_in)
    print("Truncated dat_out:", truncated_dat_out)
    
    Output:
    Truncated dat_in: [[1, 2, 3, 4], [6, 7, 8, 9], [13, 14, 15, 16]]
    Truncated dat_out: [[17, 18, 19, 20], [22, 23, 24, 25], [28, 29, 30, 31]]

    '''

    mean_in = np.median(dat_in, 1)
    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_in = np.std(dat_in)
        std_out = np.std(dat_in)
    else:
        std_in = np.std(dat_in, 1)
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        pr_in = -scipy.stats.norm.logpdf(sc, mean_in, std_in + 1e-30)
        pr_out = -scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)
        score = pr_in - pr_out

        prediction.extend(score.mean(1))
        answers.extend(ans)

    return prediction, answers


def generate_ours_offline(keep, scores, check_keep, check_scores, in_size=100000, out_size=100000,
                          fix_variance=False):
    """
    Fit a single predictive model using keep and scores in order to predict
    if the examples in check_scores were training data or not, using the
    ground truth answer from check_keep.
    """
    '''
     fits a single predictive model using the keep and scores parameters.
    '''
    dat_in = []
    dat_out = []

    for j in range(scores.shape[1]):
        dat_in.append(scores[keep[:, j], j, :])
        dat_out.append(scores[~keep[:, j], j, :])

    out_size = min(min(map(len, dat_out)), out_size)

    dat_out = np.array([x[:out_size] for x in dat_out])

    mean_out = np.median(dat_out, 1)

    if fix_variance:
        std_out = np.std(dat_out)
    else:
        std_out = np.std(dat_out, 1)

    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        score = scipy.stats.norm.logpdf(sc, mean_out, std_out + 1e-30)

        prediction.extend(score.mean(1))
        answers.extend(ans)
    return prediction, answers


def generate_global(keep, scores, check_keep, check_scores):
    """
    Use a simple global threshold sweep to predict if the examples in
    check_scores were training data or not, using the ground truth answer from
    check_keep.
    """
    prediction = []
    answers = []
    for ans, sc in zip(check_keep, check_scores):
        prediction.extend(-sc.mean(1))
        answers.extend(ans)

    return prediction, answers


def do_plot(fn, keep, scores, ntest, legend='', metric='auc', sweep_fn=sweep, **plot_kwargs):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """

    prediction, answers = fn(keep[:-ntest],
                             scores[:-ntest],
                             keep[-ntest:],
                             scores[-ntest:])

    fpr, tpr, auc, acc = sweep_fn(np.array(prediction), np.array(answers, dtype=bool))

    low = tpr[np.where(fpr < .001)[0][-1]]

    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f' % (legend, auc, acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f' % auc
    elif metric == 'acc':
        metric_text = 'acc=%.3f' % acc

    plt.plot(fpr, tpr, label=legend + metric_text, **plot_kwargs)
    return (acc, auc)


def fig_fpr_tpr():
    plt.figure(figsize=(4, 3))

    do_plot(generate_ours,
            keep, scores, 1,
            "Ours (online)\n",
            metric='auc'
            )

    do_plot(functools.partial(generate_ours, fix_variance=True),
            keep, scores, 1,
            "Ours (online, fixed variance)\n",
            metric='auc'
            )

    do_plot(functools.partial(generate_ours_offline),
            keep, scores, 1,
            "Ours (offline)\n",
            metric='auc'
            )

    do_plot(functools.partial(generate_ours_offline, fix_variance=True),
            keep, scores, 1,
            "Ours (offline, fixed variance)\n",
            metric='auc'
            )

    do_plot(generate_global,
            keep, scores, 1,
            "Global threshold\n",
            metric='auc'
            )

    plt.semilogx()
    plt.semilogy()
    plt.xlim(1e-5, 1)
    plt.ylim(1e-5, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0, 1], [0, 1], ls='--', color='gray')
    plt.subplots_adjust(bottom=.18, left=.18, top=.96, right=.96)
    plt.legend(fontsize=8)
    plt.savefig("/tmp/fprtpr.png")
    plt.show()


if __name__ == '__main__':
    load_data("exp/cifar10/")
    fig_fpr_tpr()
