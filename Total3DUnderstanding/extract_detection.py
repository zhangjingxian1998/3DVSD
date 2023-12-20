import h5py
import json
import os

vsd_data_path = '../data/vsd_boxes36.h5'
vocab_path = '../data/objects_vocab.txt'
save_root = './data_detection'
# threshold = 0.7
with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = f.read().split('\n')
    with h5py.File(vsd_data_path,'r') as g:
        vsd_data = g
        for idx,data_one in enumerate(vsd_data):
            detections = []
            boxes = vsd_data[data_one]['boxes'][()]
            obj_id = vsd_data[data_one]['obj_id'][()]
            score = vsd_data[data_one]['obj_conf'][()]
            # threshold_tmp = threshold
            # flag = True
            # while flag:
            for i in range(len(boxes)):
                # if score[i] > threshold_tmp:
                detections_one = {}
                detections_one['bbox'] = boxes[i].tolist()
                detections_one['class'] = vocab[obj_id[i]]
                detections.append(detections_one)
                # if len(detections) < 2:
                #     threshold_tmp = threshold_tmp - 0.5
                # else:
                #     flag = False
                
            save_path = os.path.join(save_root,data_one)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            filename = os.path.join(save_path,'detections.json')
            # if not os.path.exists(filename):
            with open (filename,'w') as w:
                json.dump(detections,w)
                print(idx)