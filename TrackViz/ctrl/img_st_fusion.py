#coding=utf-8
import shutil

import os

from profile.fusion_param import get_fusion_param, ctrl_msg
from train.st_estim import get_predict_delta_tracks, prepare_rand_folder, prepare_diff_folder
from train.st_filter import fusion_st_img_ranker, fusion_st_gallery_ranker

# need to run on src directory
from util.file_helper import safe_remove, safe_mkdir


def test_fusion(fusion_param, ep=0.5, en=0.01):
    # copy sort pickle
    safe_remove(fusion_param['distribution_pickle_path'])
    try:
        # 直接使用训练集的时空模型
        shutil.copy(fusion_param['src_distribution_pickle_path'], fusion_param['distribution_pickle_path'])
        print 'copy train track distribute pickle done'
    except shutil.Error:
        print 'pickle ready'
    # merge visual probability and track distribution probability
    fusion_st_gallery_ranker(fusion_param)
    # evaluate
    # todo transfer: no eval by fusion code
    # eval_on_train_test(fusion_param, test_mode=True)


def train_fusion(fusion_param, ep=0.5, en=0.01):
    # 这里不需要再做一次时空模型建立
    # get_predict_tracks(fusion_param)
    # get distribution sorted list for probability compute
    # store_sorted_deltas(fusion_param)
    fusion_st_img_ranker(fusion_param)
    # evaluate
    # todo transfer: no eval by fusion code
    # eval_on_train_test(fusion_param)


def init_strict_img_st_fusion():
    # 全局调度入口，会同时做训练集和测试集上的融合与评分
    fusion_param = get_fusion_param()
    safe_mkdir('data/' + ctrl_msg['data_folder_path'])
    get_predict_delta_tracks(fusion_param)
    # # only get rand model for train dataset
    prepare_rand_folder(fusion_param)
    prepare_diff_folder(fusion_param)

    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
    fusion_param = get_fusion_param()
    # 生成随机时空点的时空模型
    get_predict_delta_tracks(fusion_param, random=True)

    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'].replace('rand', 'diff')
    fusion_param = get_fusion_param()
    get_predict_delta_tracks(fusion_param, diff_person=True)

    # 改回非随机的train目录
    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-5]

    # has prepared more accurate ep, en
    print('fusion on training dataset')
    iter_strict_img_st_fusion(on_test=False)
    # 改成测试目录
    print('fusion on test dataset')
    ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-4] + 'est'
    safe_mkdir('data/' + ctrl_msg['data_folder_path'])
    iter_strict_img_st_fusion(on_test=True)


# def init_strict_img_st_fusion():
#     # 全局调度入口，会同时做训练集和测试集上的融合与评分
#     fusion_param = get_fusion_param()
#     print('init predict tracks into different class files')
#     # pick predict tracks into different class file
#     get_predict_tracks(fusion_param)
#     # get distribution sorted list for probability compute
#     store_sorted_deltas(fusion_param)
#
#     # # only get rand model for train dataset
#     print('generate random predict')
#     write_rand_pid(fusion_param)
#     ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
#     fusion_param = get_fusion_param()
#     # 生成随机时空点的时空模型
#     gen_rand_st_model(fusion_param)
#
#     # 改回非随机的train目录
#     ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-5]
#
#     # has prepared more accurate ep, en
#     print('fusion on training dataset')
#     iter_strict_img_st_fusion(on_test=False)
#     # 改成测试目录
#     print('fusion on test dataset')
#     ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-4] + 'est'
#     iter_strict_img_st_fusion(on_test=True)


def iter_strict_img_st_fusion(on_test=False):
    """
    call after img classifier update, train with new vision score and ep en
    :param on_test:
    :return:
    """
    fusion_param = get_fusion_param()
    # ep, en = get_shot_rate()
    if on_test:
        test_fusion(fusion_param)
    else:
        train_fusion(fusion_param)
        # update_epen(fusion_param, True)


if __name__ == '__main__':
    # img_st_fusion()
    # retrain_fusion()
    # init_strict_img_st_fusion()
    # for i in range(10):
    #     print('iteration %d' % i)
    #     ctrl_msg['cross_idx'] = i
    #     # ctrl_msg['data_folder_path'] = 'top-m2g-std%d-r-train' % i
    #     # fusion_param = get_fusion_param()
    #     # get_predict_tracks(fusion_param)
    #     # store_sorted_deltas(fusion_param)
    #     # ctrl_msg['data_folder_path'] = 'top-m2g-std%d-r-test' % i
    #     # iter_strict_img_st_fusion(on_test=True)
    #     ctrl_msg['data_folder_path'] = 'top-m2g-std%d-test' % i
    #     iter_strict_img_st_fusion(on_test=True)
    # # viz fusion curve
    # fusion_param = get_fusion_param()
    # get_predict_tracks(fusion_param)
    # store_sorted_deltas(fusion_param)
    #
    # print('generate random predict')
    # write_rand_pid(fusion_param)
    # ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'] + '_rand'
    # fusion_param = get_fusion_param()
    # gen_rand_st_model(fusion_param)
    #
    # ctrl_msg['data_folder_path'] = ctrl_msg['data_folder_path'][:-5]
    # fusion_param = get_fusion_param()

    # ctrl_msg['data_folder_path'] = 'market_market-train'
    # fusion_param = get_fusion_param()
    # init_strict_img_st_fusion()
    # ctrl_msg['data_folder_path'] = 'market_market-test'
    # fusion_param = get_fusion_param()
    # os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    # os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
    #           + 'market' + ' ' + fusion_param['eval_fusion_path'])
    #
    for i in range(0, 4):
        for j in range(0, 4 - i):
            ctrl_msg['ep'] = i * 0.25
            ctrl_msg['en'] = j * 0.25
            ctrl_msg['data_folder_path'] = 'grid_market-train'
            fusion_param = get_fusion_param()
            init_strict_img_st_fusion()
            ctrl_msg['data_folder_path'] = 'grid_market-test'
            fusion_param = get_fusion_param()
            os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
            os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
                      + 'market' + ' ' + fusion_param['eval_fusion_path'])
    # ctrl_msg['ep'] = 0.25
    # ctrl_msg['en'] = 0.5
    # ctrl_msg['data_folder_path'] = 'grid_market-train'
    # fusion_param = get_fusion_param()
    # init_strict_img_st_fusion()
    # ctrl_msg['data_folder_path'] = 'grid_market-test'
    # fusion_param = get_fusion_param()
    # os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    # os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
    #           + 'market' + ' ' + fusion_param['eval_fusion_path'])
    # ctrl_msg['ep'] = 0.5
    # ctrl_msg['en'] = 0.25
    # ctrl_msg['data_folder_path'] = 'grid_market-train'
    # fusion_param = get_fusion_param()
    # init_strict_img_st_fusion()
    # ctrl_msg['data_folder_path'] = 'grid_market-test'
    # fusion_param = get_fusion_param()
    # os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    # os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
    #           + 'market' + ' ' + fusion_param['eval_fusion_path'])
    # for cv_num in range(10):
    #     for i in range(0, 4):
    #         for j in range(0, 4 - i):
    #             ctrl_msg['ep'] = i * 0.25
    #             ctrl_msg['en'] = j * 0.25
    #             ctrl_msg['data_folder_path'] = 'market_grid-cv%d-train' % cv_num
    #             fusion_param = get_fusion_param()
    #             init_strict_img_st_fusion()
    #             ctrl_msg['data_folder_path'] = 'market_grid-cv%d-test' % cv_num
    #             fusion_param = get_fusion_param()
    #             os.environ.setdefault('LD_LIBRARY_PATH', '/usr/local/cuda/lib64')
    #             os.system('/home/cwh/anaconda2/bin/python /home/cwh/coding/rank-reid/rank_reid.py 2 '
#                       + ('grid-cv%d' % cv_num) + ' ' + fusion_param['eval_fusion_path'])
    # delta_range, raw_probs, rand_probs, over_probs = fusion_curve(fusion_param)
    # viz_fusion_curve(delta_range, [raw_probs, rand_probs, over_probs])

    # viz smooth dist
    # viz_market_distribution(fusion_param)
