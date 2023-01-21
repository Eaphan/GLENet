
import pickle
import numpy as np
from tqdm import tqdm

# 1. read uncertainty prediciton
file_path = 'output/uncertainty_dump/waymo_un_v3.pkl'
with open(file_path, 'rb') as f:
    uncertainty_map = pickle.load(f)

tfrecord_list = open('waymo/ImageSets/train.txt').readlines()
tfrecord_list =[x.strip() for x in tfrecord_list]

for tfrecord in tqdm(tfrecord_list):
    item = tfrecord.replace('.tfrecord', '')
    info_path = f'waymo/waymo_processed_data_v0_5_0/{item}/{item}.pkl'
    with open(info_path, 'rb') as f:
        t_infos = pickle.load(f)

    for info in t_infos:
        # import pdb;pdb.set_trace()
        frame_id = info['point_cloud']['lidar_sequence'] + "#" + str(info['point_cloud']['sample_idx'])
        # index_list = info['annos']['index']
        names = info['annos']['name']
        # import pdb;pdb.set_trace()
        uncertainty_list = []
        for idx in range(len(names)):
            name = names[idx]
            if name!='Vehicle':
                uncertainty = np.array([-1 for i in range(7)])
            else:
                key = frame_id + '_' + str(idx)
                uncertainty = uncertainty_map[key]
            uncertainty_list.append(uncertainty)
        # import pdb;pdb.set_trace()
        if len(uncertainty_list)==0:
            uncertainty_list = np.zeros([0, 7])
        info['annos']['uncertainty'] = np.array(uncertainty_list)

    with open(info_path, 'wb') as f:
        pickle.dump(t_infos, f)

# 3. change waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl
file_path = 'waymo/waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl'
with open(file_path, 'rb') as f:
    db_infos = pickle.load(f)

for info in tqdm(db_infos['Vehicle']):
    frame_id = info['sequence_name'] + "#" + str(info["sample_idx"])
    gt_idx = info['gt_idx']
    key = frame_id + '_' + str(gt_idx)
    uncertainty = uncertainty_map[key]
    info['uncertainty'] = uncertainty


file_path = 'waymo/waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1_wconf_v3.pkl'
with open(file_path, 'wb') as f:
    pickle.dump(db_infos, f)

exit(0)


