def folder(path):
    final_slash_idx = -1
    for i, c in enumerate(path):
        if c == '/':
            final_slash_idx = i
    if final_slash_idx == -1:
        return path
    else:
        return path[: final_slash_idx]


if __name__ == '__main__':
    print(folder('data/top10/test.txt'))
