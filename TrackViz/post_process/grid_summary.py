from util.file_helper import read_lines, write


def avg_acc(grid_eval_path):
    grid_infos = read_lines(grid_eval_path)
    before_vision_accs = [0.0, 0.0, 0.0]
    before_fusion_accs = [0.0, 0.0, 0.0]
    after_vision_accs = [0.0, 0.0, 0.0]
    after_fusion_accs = [0.0, 0.0, 0.0]
    i_cv_cnt = 0
    for i, grid_info in enumerate(grid_infos):
        if i % 2 != 0:
            accs = grid_info.split()
            if i_cv_cnt % 4 == 0:
                for j in range(3):
                    before_vision_accs[j] += float(accs[j])
            if i_cv_cnt % 4 == 1:
                for j in range(3):
                    before_fusion_accs[j] += float(accs[j])
            if i_cv_cnt % 4 == 2:
                for j in range(3):
                    after_vision_accs[j] += float(accs[j])
            if i_cv_cnt % 4 == 3:
                for j in range(3):
                    after_fusion_accs[j] += float(accs[j])
            i_cv_cnt += 1
    write('grid_eval.log', '\n' + grid_eval_path + '\n')
    write('grid_eval.log', 'before_retrain_vision\n%f %f %f\n' % (before_vision_accs[0]/10, before_vision_accs[1]/10, before_vision_accs[2]/10))
    write('grid_eval.log', 'before_retrain_fusion\n%f %f %f\n' % (before_fusion_accs[0]/10, before_fusion_accs[1]/10, before_fusion_accs[2]/10))
    write('grid_eval.log', 'after_retrain_vision\n%f %f %f\n' % (after_vision_accs[0]/10, after_vision_accs[1]/10, after_vision_accs[2]/10))
    write('grid_eval.log', 'after_retrain_fusion\n%f %f %f\n' % (after_fusion_accs[0]/10, after_fusion_accs[1]/10, after_fusion_accs[2]/10))


if __name__ == '__main__':
    avg_acc('market_grid.txt')
    avg_acc('cuhk_grid.txt')
    avg_acc('viper_grid.txt')
    avg_acc('grid_grid.txt')