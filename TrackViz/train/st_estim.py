#coding=utf-8
from random import randint

import shutil

from profile.fusion_param import ctrl_msg
from util.file_helper import read_lines, safe_remove, safe_mkdir
from util.serialize import pickle_save
from util.str_helper import folder


def prepare_rand_folder(fusion_param):
    rand_predict_path = fusion_param['renew_pid_path'].replace(ctrl_msg['data_folder_path'],
                                                           ctrl_msg['data_folder_path'] + '_rand')
    rand_folder_path = folder(rand_predict_path)
    safe_mkdir(rand_folder_path)
    # although copy all info including pid info, but not use in later training
    shutil.copy(fusion_param['renew_pid_path'], rand_predict_path)


def prepare_diff_folder(fusion_param):
    diff_predict_path = fusion_param['renew_pid_path'].replace(ctrl_msg['data_folder_path'],
                                                           ctrl_msg['data_folder_path'] + '_diff')
    diff_folder_path = folder(diff_predict_path)
    safe_mkdir(diff_folder_path)
    # although copy all info including pid info, but not use in later training
    shutil.copy(fusion_param['renew_pid_path'], diff_predict_path)


def get_predict_delta_tracks(fusion_param, useful_predict_limit=10, random=False, diff_person=False, use_real_st=False):
    # 获取左图列表
    answer_path = fusion_param['answer_path']
    answer_lines = read_lines(answer_path)
    camera_cnt = 6
    real_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            #
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            # grid
            real_tracks.append([info[0], int(info[1][0]), int(info[2]), 1])
        elif 'f' in info[2]:
            real_tracks.append([info[0], int(info[1][1]), int(info[2][1:-5]), 1])
            camera_cnt = 8
        else:
            # market
            real_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
    print 'left image ready'
    # 获取右图列表
    renew_pid_path = fusion_param['renew_pid_path']
    predict_lines = read_lines(renew_pid_path)
    print 'predict images ready'

    # 左图中的人在右图可能出现在6个摄像头中
    camera_delta_s = [[list() for j in range(camera_cnt)] for i in range(camera_cnt)]
    person_cnt = len(answer_lines)
    # market1501数据集有六个序列，只有同一个序列才能计算delta
    if random:
        useful_predict_limit = max(len(predict_lines)/100, 10)
    for i, line in enumerate(predict_lines):
        predict_pids = line.split(' ')
        useful_cnt = 0
        for j, predict_pid in enumerate(predict_pids):
            if useful_cnt > useful_predict_limit:
                break
            if random:
                predict_pid = randint(0, person_cnt - 1)
            elif diff_person:
                predict_pid = randint(10, person_cnt - 1)
            else:
                # todo transfer: if predict by python, start from 0, needn't minus 1
                predict_pid = int(predict_pid)
            predict_pid = int(predict_pid)
            # same seq
            # todo ignore same camera track
            if real_tracks[i][3] == real_tracks[predict_pid][3] and real_tracks[i][1] != real_tracks[predict_pid][1]:
                # and pid equal: real st
                # if use_real_st and random or real_tracks[i][0] == real_tracks[predict_pid][0]:
                if True:
                    useful_cnt += 1
                    delta = real_tracks[i][2] - real_tracks[predict_pid][2]
                    if abs(delta) < 1000000:
                        camera_delta_s[real_tracks[i][1] - 1][real_tracks[predict_pid][1] - 1].append(delta)
    print 'deltas collected'
    for camera_delta in camera_delta_s:
        for delta_s in camera_delta:
            delta_s.sort()
    print 'deltas sorted'
    # for python
    safe_remove(fusion_param['distribution_pickle_path'])
    pickle_save(fusion_param['distribution_pickle_path'], camera_delta_s)
    print 'deltas saved'
    return camera_delta_s