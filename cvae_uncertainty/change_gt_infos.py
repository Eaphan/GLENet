# kitti_infos_train.pkl
# kitti_dbinfos_train.pkl

import pickle
import numpy as np

# 1. read uncertainty prediciton
file_path = 'output/uncertainty_dump/un_v4.pkl'
with open(file_path, 'rb') as f:
    uncertainty_map = pickle.load(f)

# 2. change kitti_infos_train.pkl
file_path = 'kitti/kitti_infos_train_ori.pkl'
with open(file_path, 'rb') as f:
    kitti_infos = pickle.load(f)

for info in kitti_infos:
    # import pdb;pdb.set_trace()
    frame_id = info['image']['image_idx']
    index_list = info['annos']['index']
    names = info['annos']['name']
    # import pdb;pdb.set_trace()
    uncertainty_list = []
    for i,idx in enumerate(index_list):
        name = names[i]
        if name!='Car':
            uncertainty = np.array([-1 for i in range(7)])
        else:
            key = frame_id + '_' + str(idx)
            uncertainty = uncertainty_map[key]
        uncertainty_list.append(uncertainty)
    # import pdb;pdb.set_trace()
    info['annos']['uncertainty'] = np.array(uncertainty_list)

file_path = 'kitti/kitti_infos_train_wconf_v4.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(kitti_infos, f)


# 3. change kitti_dbinfos_train.pkl
file_path = 'kitti/kitti_dbinfos_train_ori.pkl'
with open(file_path, 'rb') as f:
    db_infos = pickle.load(f)

for info in db_infos['Car']:
    frame_id = info['image_idx']
    gt_idx = info['gt_idx']
    key = frame_id + '_' + str(gt_idx)
    uncertainty = uncertainty_map[key]
    info['uncertainty'] = uncertainty


file_path = 'kitti/kitti_dbinfos_train_wconf_v4.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(db_infos, f)
