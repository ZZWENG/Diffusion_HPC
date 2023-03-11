import os
import pickle
from tqdm import tqdm

path = './data/eft_smart'
actions = os.listdir(path)
print(actions)
for action in actions:
    action_path = os.path.join(path, action)
    files = os.listdir(action_path)
    for file in tqdm(files, desc=action):
        with open(os.path.join(action_path, file), 'rb') as f:
            data = pickle.load(f)
        data['imageName'] = [os.path.join(
            './data', 
            'SportsCap_Dataset_SMART_v1/images',
            os.path.basename(data['imageName'][0])
        )]
        with open(os.path.join(action_path, file), 'wb') as f:
            pickle.dump(data, f)
