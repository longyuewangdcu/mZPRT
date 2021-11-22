import sys 
import jieba

def jieba_cws(string):
    seg_list = jieba.cut(string.strip())
    return ' '.join(seg_list)


if __name__ == '__main__':
    
    jieba.load_userdict('./userdict.txt')
    with sys.stdin as f:
        for line in f:
            line_cws = jieba_cws(line)
            sys.stdout.write(line_cws.strip())
            sys.stdout.write('\n')
