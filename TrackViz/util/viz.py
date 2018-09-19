import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def draw_line(y_s, x_s, y_label, x_label, y_titles, title, line_color=None):
    # plt.subplots()
    plt.subplots(figsize=(6, 5))
    sns.set(font_scale=2.4)
    line_styles = ['--', '-']
    for i in range(len(y_s)):
        plt.plot(x_s, y_s[i], color=line_color, label=y_titles[i], linestyle=line_styles[i], linewidth=5.)
    plt.xlabel(x_label, fontsize=32)
    plt.ylabel(y_label, fontsize=32)
    plt.ylim(min(min(y_s[0]), min(y_s[1]))*0.8, max(max(y_s[0]), max(y_s[1]))*1.2)
    plt.xlim(min(x_s), max(x_s))
    plt.yticks(fontsize=32)
    plt.xticks(fontsize=32)
    plt.legend()
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    accs = np.genfromtxt('../increment_acc.txt', delimiter='\t')
    draw_line(accs, np.arange(1, 11), 'Rank1_acc', 'iteration times',['vision', 'fusion'],  title='')
    # plt.subplots()
    # plt.plot(np.arange(1, 11), accs[0], label='vision')
    # plt.legend()
    # plt.plot(np.arange(1, 11), accs[0], label='fusion')
    # plt.legend()
    # plt.xlabel('Rank1_acc')
    # plt.ylabel('iteration times')
    # plt.ylim(0.2, 0.4)
    # plt.title('')
    # plt.show()