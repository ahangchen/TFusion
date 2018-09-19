import os


def write_line(path, content):
    with open(path, "a+") as dst_file:
        dst_file.write(content + '\n')


def write(path, content):
    with open(path, "a+") as dst_file:
        dst_file.write(content)


def read_lines(path):
    with open(path) as f:
        content = list()
        while 1:
            try:
                lines = f.readlines(100)
            except UnicodeDecodeError:
                f.close()
                continue
            if not lines:
                break
            for line in lines:
                content.append(line)
    return content


def read_lines_and(path, on_line):
    with open(path) as f:
        content = list()
        while 1:
            try:
                lines = f.readlines(100)
            except UnicodeDecodeError:
                f.close()
                continue
            if not lines:
                break
            for line in lines:
                on_line(line)
    return content


def read_lines_idx_and(path, on_line):
    line_idx = 0
    with open(path) as f:
        content = list()
        while 1:
            try:
                lines = f.readlines(100)
            except UnicodeDecodeError:
                f.close()
                continue
            if not lines:
                break
            for line in lines:
                on_line(line, line_idx)
                line_idx += 1
    return content


def safe_remove(path):
    if os.path.exists(path):
        os.remove(path)
        return True
    else:
        return False


def safe_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
