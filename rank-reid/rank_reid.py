import sys

from baseline.evaluate import market_result_eval, grid_result_eval
from pretrain.eval import train_pair_predict,test_pair_predict, train_rank_predict, test_rank_predict
from transfer.simple_rank_transfer import rank_transfer_2dataset


def get_source_target_info(source, target):
    source_model_path = '/home/cwh/coding/rank-reid/pretrain/%s_pair_pretrain.h5' % source
    target_dataset_path = ''
    if target == 'market':
        target_dataset_path = '/home/cwh/coding/Market-1501'
    elif target == 'markets1':
        target_dataset_path = '/home/cwh/coding/markets1'
    elif target == 'duke':
        target_dataset_path = '/home/cwh/coding/DukeMTMC-reID'
    elif 'grid' in target:
        target_dataset_path = '/home/cwh/coding/grid_train_probe_gallery' + target.replace('grid-cv', '/cross')
    return source_model_path, target_dataset_path


def vision_predict(source, target, train_pid_path, train_score_path, test_pid_path, test_score_path):
    source_model_path, target_dataset_path = get_source_target_info(source, target)
    target_probe_path = target_dataset_path + '/probe'
    target_train_path = target_dataset_path + '/train'
    target_gallery_path = target_dataset_path + '/test'
    train_pair_predict(source_model_path, target_train_path, train_pid_path, train_score_path)
    test_pair_predict(source_model_path, target_probe_path, target_gallery_path, test_pid_path, test_score_path)
    predict_eval(target, test_pid_path)


def rank_transfer(source, target, target_train_list, fusion_train_rank_pids_path, fusion_train_rank_scores_path):
    source_model_path, target_dataset_path = get_source_target_info(source, target)
    target_train_path = target_dataset_path + '/train'
    target_model_path = source + '_' + target + '-rank_transfer.h5'
    rank_transfer_2dataset(source_model_path, target_train_list, target_model_path, target_train_path,
                           fusion_train_rank_pids_path, fusion_train_rank_scores_path)
    return target_model_path


def rank_predict(rank_model_path, target, transfer_train_rank_pids_path, transfer_train_rank_scores_path,
                    transfer_test_rank_pids_path, transfer_test_rank_scores_path):
    source_model_path, target_dataset_path = get_source_target_info(source, target)
    target_train_path = target_dataset_path + '/train'
    target_probe_path = target_dataset_path + '/probe'
    target_gallery_path = target_dataset_path + '/test'
    # train_rank_predict(rank_model_path, target_train_path, transfer_train_rank_pids_path, transfer_train_rank_scores_path)
    test_rank_predict(rank_model_path, target_probe_path, target_gallery_path, transfer_test_rank_pids_path, transfer_test_rank_scores_path)
    predict_eval(target, transfer_test_rank_pids_path)


def predict_eval(target, predict_path):
    if target == 'market' or target == 'market-r':
        market_result_eval(predict_path,
                           TEST = '/home/cwh/coding/Market-1501/test', QUERY = '/home/cwh/coding/Market-1501/probe')
    elif 'grid' in target:
        grid_result_eval(predict_path)
    elif 'duke' in target:
        market_result_eval(predict_path, log_path='duke_eval.log', TEST = '/home/cwh/coding/DukeMTMC-reID/test', QUERY = '/home/cwh/coding/DukeMTMC-reID/probe')


if __name__ == '__main__':
    # source = 'cuhk'
    # target = 'market'
    # fusion_train_rank_pids_path = '/home/cwh/coding/TrackViz/data/%s_%s-train/cross_filter_pid.log' % (source, target)
    # fusion_train_rank_scores_path = '/home/cwh/coding/TrackViz/data/%s_%s-train/cross_filter_score.log' % (source, target)
    # transfer_train_rank_pids_path = 'train_rank_pid.log'
    # transfer_train_rank_scores_path = 'train_rank_score.log'
    # transfer_test_rank_pids_path = 'test_rank_pid.log'
    # transfer_test_rank_scores_path = 'test_rank_score.log'
    # target_train_list ='dataset/market_train.list'
    # rank_model_path = rank_transfer(source, target, target_train_list, fusion_train_rank_pids_path,
    #                                fusion_train_rank_scores_path)
    # # rank_model_path = '/home/cwh/coding/rank-reid/i_' + source + '_' + target + '-rank_transfer.h5'
    # # rank_model_path = 'transfer/rank_transfer_test.h5'
    # rank_predict(rank_model_path, target, transfer_train_rank_pids_path, transfer_train_rank_scores_path,
    #              transfer_test_rank_pids_path, transfer_test_rank_scores_path)

    opt = sys.argv[1]
    if opt == '0':
        source = sys.argv[2]
        target = sys.argv[3]
        vision_train_rank_pids_path = sys.argv[4]
        vision_train_rank_scores_path = sys.argv[5]
        vision_test_rank_pids_path = sys.argv[6]
        vision_test_rank_scores_path = sys.argv[7]
        vision_predict(source, target,
                       vision_train_rank_pids_path, vision_train_rank_scores_path,
                       vision_test_rank_pids_path, vision_test_rank_scores_path)
    elif opt == '1':
        source = sys.argv[2]
        target = sys.argv[3]
        fusion_train_rank_pids_path = sys.argv[4]
        fusion_train_rank_scores_path = sys.argv[5]
        transfer_train_rank_pids_path = sys.argv[6]
        transfer_train_rank_scores_path = sys.argv[7]
        transfer_test_rank_pids_path = sys.argv[8]
        transfer_test_rank_scores_path = sys.argv[9]
        target_train_list = sys.argv[10]
        rank_model_path = rank_transfer(source, target, target_train_list, fusion_train_rank_pids_path, fusion_train_rank_scores_path)
        rank_predict(rank_model_path, target, transfer_train_rank_pids_path, transfer_train_rank_scores_path,
                    transfer_test_rank_pids_path, transfer_test_rank_scores_path)
    elif opt == '2':
        target = sys.argv[2]
        predict_path = sys.argv[3]
        predict_eval(target, predict_path)

    else:
        pass
