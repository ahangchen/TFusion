# coding=utf-8
from util.serialize import pickle_load


def binary_search(a, target):
    # 不同于普通的二分查找，目标是寻找target最适合的index
    low = 0
    high = len(a) - 1

    while low <= high:
        mid = (low + high) // 2
        mid_val = a[mid]
        if mid_val < target:
            low = mid + 1
        elif mid_val > target:
            high = mid - 1
        else:
            return mid
    return low


def track_score(camera_delta_s, camera1, time1, camera2, time2, interval=100, test=True, filter_interval=1000):
    if abs(time1 - time2) > filter_interval:
        return -1.
    camera1 -= 1
    camera2 -= 1
    # if test and camera1 == camera2:
    #     return 0.0000001
    cur_delta = time1 - time2
    delta_distribution = camera_delta_s[camera1][camera2]
    total_cnt = sum(map(len, camera_delta_s[camera1]))
    # 10 second
    left_bound = cur_delta - interval
    right_bound = cur_delta + interval
    # 二分查找位置，得到容错区间内时空点数量
    left_index = binary_search(delta_distribution, left_bound)
    right_index = binary_search(delta_distribution, right_bound)
    if total_cnt == 0 or len(camera_delta_s[camera1][camera2]) == 0:
        return 0.0
    # 这里除以total_cnt而非len(camera_delta_s[camera1][camera2])，体现空间概率
    score = (right_index - left_index) / float(total_cnt)
    # 训练集中同摄像头概率很高,但评估又不要同摄像头的,体现空间概率很不划算
    # score = (right_index - left_index + 1) / float(len(camera_delta_s[camera1][camera2]))
    if len(delta_distribution) == 0:
        return 0.0
    # score = (right_index - left_index + 1) / float(len(camera_delta_s[camera1][2]))
    # if score > 0:
    #     print(len(delta_distribution))
    #     print('delta range %d ~ %d' % (delta_distribution[0], delta_distribution[-1]))
    #     print(left_index)
    #     print(right_index)
    #     print('probablity: %f%%' % (score * 100))
    return score


def track_interval_score(interval_score_s, camera1, time1, camera2, time2):
    delta = time2 - time1
    for i, camera_pair_travel_prob in enumerate(interval_score_s[camera1 - 1][camera2 - 1]):
        if camera_pair_travel_prob['left'] < delta < camera_pair_travel_prob['right']:
            print('camera1: %d, camera2: %d, delta: %d, interval: %d, prob: %f' % (
                camera1, camera2, delta, i, camera_pair_travel_prob['prob']))
            return camera_pair_travel_prob['prob']
    return 0

if __name__ == '__main__':
    camera_delta_s = pickle_load('data/top10/sorted_deltas.pickle')
    track_score(camera_delta_s, 1, 25, 2, 250)