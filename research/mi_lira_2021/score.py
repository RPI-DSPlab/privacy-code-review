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

import sys
import numpy as np
import os
import multiprocessing as mp


def load_one(base):
    """
    This loads a  logits and converts it to a scored prediction.
    """
    print(f"-----Start load_one, root: {os.path.join(logdir, base, 'logits')}-----")
    root = os.path.join(logdir, base, 'logits')
    if not os.path.exists(root): return None

    if not os.path.exists(os.path.join(logdir, base, 'scores')):
        os.mkdir(os.path.join(logdir, base, 'scores'))

    for f in os.listdir(root):
        try:
            opredictions = np.load(os.path.join(root, f))
        except:
            print("Fail")
            continue

        ## Be exceptionally careful.
        ## Numerically stable everything, as described in the paper.
        predictions = opredictions - np.max(opredictions, axis=3, keepdims=True)
        predictions = np.array(np.exp(predictions), dtype=np.float64)
        predictions = predictions / np.sum(predictions, axis=3, keepdims=True)

        """
        The predictions array that you've obtained is the output of your models for each example in your dataset. 
        The shape of the array is (50000, 1, 2, 10), which means that there are 50,000 examples, each with 1 output 
        vector, and each output vector contains 2 augmentations of the 10 class probabilities (logits). Here is a 
        breakdown of what each dimension represents:

        50000: This is the number of examples in your dataset. Each example is being processed separately, hence why 
        this is the first dimension of the array.
        
        1: This represents the output vector for each example. In this case, it seems like you're only using a single 
        output vector for each example.
        
        2: This represents the number of augmentations for each example. Data augmentation is a technique used to 
        increase the diversity of your training set by applying random (but realistic) transformations such as 
        rotations, translations, flips, etc. This helps to make the model more robust to variations in the input data.
        
        10: This represents the number of classes in your dataset. Each value in this last dimension of the array 
        represents the model's predicted probability (logit) for each class.
        
        The values in the predictions array are probabilities (or logits, depending on whether a softmax function has 
        been applied) that the model assigns to each class. The higher the value, the more confident the model is that 
        the example belongs to that class. 
        """

        COUNT = predictions.shape[0]
        #  x num_examples x num_augmentations x logits
        y_true = predictions[np.arange(COUNT), :, :, labels[:COUNT]]
        print(y_true.shape)

        print('mean acc', np.mean(predictions[:, 0, 0, :].argmax(1) == labels[:COUNT]))

        predictions[np.arange(COUNT), :, :, labels[:COUNT]] = 0
        y_wrong = np.sum(predictions, axis=3)

        logit = (np.log(y_true.mean((1)) + 1e-45) - np.log(y_wrong.mean((1)) + 1e-45))

        print(f"Output dir: {os.path.join(logdir, base, 'scores', f)}")
        np.save(os.path.join(logdir, base, 'scores', f), logit)


def load_stats():
    print(f"logdir list: {os.listdir(logdir)}")
    with mp.Pool(2) as p:
        p.map(load_one, [x for x in os.listdir(logdir) if 'exp' in x])


logdir = ".\\exp\\cifar10"  # sys.argv[1]
labels = np.load(os.path.join(logdir, "y_train.npy"))

if __name__ == '__main__':
    load_stats()

