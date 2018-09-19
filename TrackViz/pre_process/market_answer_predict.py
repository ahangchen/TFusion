from profile.fusion_param import ctrl_msg, get_fusion_param
from train.st_filter import train_tracks
from util.file_helper import read_lines

import numpy as np

from util.serialize import pickle_save


def save_market_train_truth():
    ctrl_msg['data_folder_path'] = 'market_market-train'
    fusion_param = get_fusion_param()
    market_train_tracks = train_tracks(fusion_param)
    deltas = [[list() for j in range(6)] for i in range(6)]

    for i, market_train_track in enumerate(market_train_tracks):
        for j in range(max(0, i - 50), min(i + 60, len(market_train_tracks))):
            if market_train_tracks[i][0] == market_train_tracks[j][0] \
                    and i != j \
                    and market_train_tracks[i][3] == market_train_tracks[j][3] \
                    and market_train_tracks[i][1] != market_train_tracks[j][1]:
                deltas[market_train_tracks[i][1] - 1][market_train_tracks[j][1] - 1].append(
                    market_train_tracks[i][2] - market_train_tracks[j][2]
                )
    for camera_delta in deltas:
        for delta_s in camera_delta:
            delta_s.sort()
    pickle_save('true_market_train.pck', deltas)


def save_market_test_truth():
    ctrl_msg['data_folder_path'] = 'market_market-test'
    fusion_param = get_fusion_param()
    answer_path = fusion_param['answer_path']
    answer_lines = read_lines(answer_path)
    query_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            query_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            query_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])

    gallery_path = fusion_param['gallery_path']
    gallery_lines = read_lines(gallery_path)
    gallery_tracks = list()
    for gallery in gallery_lines:
        info = gallery.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            gallery_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            gallery_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
    gallery_tracks.extend(query_tracks)
    print(len(gallery_tracks))
    deltas = [[list() for j in range(6)] for i in range(6)]

    for i, market_probe_track in enumerate(gallery_tracks):
        if gallery_tracks[i][0] == 0 or gallery_tracks[i][0] == -1:
            continue
        for j in range(len(gallery_tracks)):
            if gallery_tracks[i][0] == gallery_tracks[j][0] \
                    and i != j \
                    and gallery_tracks[i][3] == gallery_tracks[j][3] \
                    and gallery_tracks[i][1] != gallery_tracks[j][1]:
                if gallery_tracks[i][1] == 4 and gallery_tracks[j][1] - 1 == 5:
                    if j >= 19732:
                        print gallery_tracks[i][2] - gallery_tracks[j][2]
                deltas[gallery_tracks[i][1] - 1][gallery_tracks[j][1] - 1].append(
                    gallery_tracks[i][2] - gallery_tracks[j][2]
                )
    for camera_delta in deltas:
        for delta_s in camera_delta:
            delta_s.sort()
    pickle_save('true_market_pg.pck', deltas)



def save_market_probe_truth():
    ctrl_msg['data_folder_path'] = 'market_market-test'
    fusion_param = get_fusion_param()
    answer_path = fusion_param['answer_path']
    answer_lines = read_lines(answer_path)
    query_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            query_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            query_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])

    gallery_path = fusion_param['gallery_path']
    gallery_lines = read_lines(gallery_path)
    gallery_tracks = list()
    for gallery in gallery_lines:
        info = gallery.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            gallery_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            gallery_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
    deltas = [[list() for j in range(6)] for i in range(6)]

    for i, market_probe_track in enumerate(query_tracks):
        if query_tracks[i][0] == 0 or gallery_tracks[i][0] == -1:
            continue
        for j in range(len(gallery_tracks)):
            if query_tracks[i][0] == gallery_tracks[j][0] \
                    and i != j \
                    and query_tracks[i][3] == gallery_tracks[j][3] \
                    and query_tracks[i][1] != gallery_tracks[j][1]:
                deltas[query_tracks[i][1] - 1][gallery_tracks[j][1] - 1].append(
                    query_tracks[i][2] - gallery_tracks[j][2]
                )
    for camera_delta in deltas:
        for delta_s in camera_delta:
            delta_s.sort()
    pickle_save('true_market_probe.pck', deltas)


def save_market_img_list(img_list_path, dest_path):
    answer_lines = read_lines(img_list_path)
    query_tracks = list()
    for i, answer in enumerate(answer_lines):
        info = answer.split('_')
        query_tracks.append([i, info[0], int(info[1][1]), int(info[2]), int(info[1][3])])
    query_tracks = np.array(query_tracks).astype(int)
    np.savetxt(dest_path, query_tracks, fmt='%d', delimiter='\t')


def save_grid_train_truth():
    ctrl_msg['data_folder_path'] = 'market_grid-cv0-train'
    fusion_param = get_fusion_param()
    market_train_tracks = train_tracks(fusion_param)
    deltas = [[list() for j in range(6)] for i in range(6)]

    for i, market_train_track in enumerate(market_train_tracks):
        for j in range(0, len(market_train_tracks)):
            if market_train_tracks[i][0] == market_train_tracks[j][0] \
                    and i != j \
                    and market_train_tracks[i][1] != market_train_tracks[j][1]:
                deltas[market_train_tracks[i][1] - 1][market_train_tracks[j][1] - 1].append(
                    market_train_tracks[i][2] - market_train_tracks[j][2]
                )
    for camera_delta in deltas:
        for delta_s in camera_delta:
            delta_s.sort()
    pickle_save('true_grid-cv0_train.pck', deltas)



def save_grid_test_truth():
    ctrl_msg['data_folder_path'] = 'market_grid-cv0-test'
    fusion_param = get_fusion_param()
    answer_path = fusion_param['answer_path']
    answer_lines = read_lines(answer_path)
    query_tracks = list()
    for answer in answer_lines:
        info = answer.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            query_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            query_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])

    gallery_path = fusion_param['gallery_path']
    gallery_lines = read_lines(gallery_path)
    gallery_tracks = list()
    for gallery in gallery_lines:
        info = gallery.split('_')
        if 'bmp' in info[2]:
            info[2] = info[2].split('.')[0]
        if len(info) > 4 and 'jpe' in info[6]:
            gallery_tracks.append([info[0], int(info[1][0]), int(info[2])])
        else:
            gallery_tracks.append([info[0], int(info[1][1]), int(info[2]), int(info[1][3])])

    deltas = [[list() for j in range(6)] for i in range(6)]

    for i, market_probe_track in enumerate(query_tracks):
        for j in range(len(gallery_tracks)):
            if query_tracks[i][0] == gallery_tracks[j][0] \
                    and i != j \
                    and query_tracks[i][1] != gallery_tracks[j][1]:
                deltas[query_tracks[i][1] - 1][gallery_tracks[j][1] - 1].append(
                    query_tracks[i][2] - gallery_tracks[j][2]
                )
    for camera_delta in deltas:
        for delta_s in camera_delta:
            delta_s.sort()
    pickle_save('true_grid-cv0_test.pck', deltas)


if __name__ == '__main__':
    # probe_path = '../data/market/probe.txt'
    # gallery_path = '../data/market/gallery.txt'
    save_grid_train_truth()
    save_grid_test_truth()
    # save_market_img_list(probe_path, 'market_probe.csv')
    # save_market_img_list(gallery_path, 'market_gallery.csv')
    # save_market_train_truth()
    # save_market_probe_truth()
    # save_market_test_truth()