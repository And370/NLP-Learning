# -*- coding: utf-8 -*-
# author:And370
# time:2020/3/1

from collections import Counter

import jieba
import numpy as np


class SimilarityModel(object):
    def __init__(self, documents_raw, mode="tf_idf"):
        """
        :param documents_raw: 原始文档
        :param mode: 默认tf_idf，可选["default", "tf_idf", "bm25", "jaccard","cos"]
        """
        # 原始文档
        self.documents_raw = documents_raw
        # 文档被jieba切分的词
        self.documents_list = [list(jieba.cut(doc)) for doc in documents_raw]
        # 文档总个数
        self.documents_length = len(documents_raw)

        # 存储每个文本中每个词的词频
        self.tfs = []
        # 存储每个词汇的文档频数
        self.df = {}
        # 存储每个词汇的逆文档频率
        self.idf = {}
        # 默认的相似度算法
        self.mode = mode

        # 类初始化
        self.init()

    def init(self):
        # 遍历 文档切分词集
        for words in self.documents_list:
            # 相对词频以per累积
            tf = {}
            per = 1 / len(words)
            for word in words:
                tf[word] = tf.get(word, 0) + per
            # 存储每个文档中每个词的相对词频
            self.tfs.append(tf)
            # count包含该词汇的文本个数
            for key in tf.keys():
                self.df[key] = self.df.get(key, 0) + 1
        for key, value in self.df.items():
            # 每个词的逆文档频率
            self.idf[key] = np.log(self.documents_length / (value + 1))

    def tf_idf(self, sentence_basic, sentence_new):
        """
        通过tf_idf计算文档间相似度
        此处是对待识别文档的所有词语的tf-idf求和
        :param sentence_basic: 文档1(句库内)
        :param sentence_new: 文档2(待识别)
        :return: 相似度分数
        """
        score = 0.0

        for q in sentence_new:
            if q not in sentence_basic:
                continue
            score += sentence_basic[q] * self.idf[q]
        return score

    def bm25(self, sentence_basic, sentence_new):
        """
        通过tf_idf计算文档间相似度
        :param sentence_basic: 文档1(句库内)
        :param sentence_new: 文档2(待识别)
        :return: 相似度分数
        """
        score = 0.0

        for q in sentence_new:
            if q not in sentence_basic:
                continue
            score += sentence_basic[q] * self.idf[q]
        return score

    def jaccard(self, sentence_basic, sentence_new):
        """
        通过jaccard计算文档间相似度
        :param sentence_basic: 文档1(句库内)
        :param sentence_new: 文档2(待识别)
        :return: 相似度分数
        """
        intersection = len(set(sentence_basic.keys()) & set(sentence_new.keys()))
        union = len(set(sentence_basic.keys()) | set(sentence_new.keys()))
        penalty_ignore_p = np.log(abs(len(sentence_basic) - len(sentence_new)) + 1)
        return intersection / (union + penalty_ignore_p)

    def cosine(self, sentence_basic, sentence_new):
        tf_new = Counter(sentence_new)
        tf_basic = Counter(sentence_basic)
        tf_both = tf_basic + tf_new
        tf_new_list = []
        tf_basic_list = []
        for key in tf_both:
            tf_new_list.append(tf_new.get(key, 0))
            tf_basic_list.append(tf_basic.get(key, 0))
        return np.dot(np.array(tf_new_list), np.array(tf_basic_list)) / \
               (np.sqrt(sum([np.square(i) for i in tf_new_list])) * np.sqrt(sum([np.square(i) for i in tf_basic_list])))

    def get_score(self, index, sentence, mode="default"):
        """
        :param index: 各文档词频索引
        :param sentence: 待识别的文档(词频形式)
        :param mode: 相似度算法
        :return: float 相似度分数
        """
        mode = mode.lower().replace("-", "_")
        if mode not in ("default", "tf_idf", "bm25", "jaccard","cos"):
            raise Exception("""parameter 'mode' should in ("default","tf_idf","bm25","jaccard","cos").""")
        if mode == "default":
            mode = self.mode
        if mode == "tf_idf":
            func = self.tf_idf
        if mode == "bm25":
            func = self.bm25
        if mode == "jaccard":
            func = self.jaccard
        if mode == "cos" or mode == "cosine":
            func = self.cosine

        return func(self.tfs[index], sentence)

    def get_scores(self, sentence, mode="default"):
        """
        调用get_score,对所有句库内文档进行相似度打分
        :param sentence: 待识别的文档(词频形式)
        :param mode: 相似度算法
        :return:
            list: 相似度分数组
        """
        score_dict = {}
        for i in range(self.documents_length):
            score = self.get_score(i, sentence, mode=mode)
            if score:
                score_dict.update({score: self.documents_raw[i]})
        return score_dict

    def get_max_score(self, sentence, mode="default"):
        """
        获取相似度最高的句库文档及分数
        :param sentence: 待识别的文档(词频形式)
        :param mode: 相似度算法
        :return:
            dict:{相似度最高分:对应文档}
        """
        score_dict = self.get_scores(sentence, mode)
        score_max = max(score_dict.keys())
        return {score_max: score_dict.get(score_max)}


if __name__ == '__main__':
    document_list = ["好饿啊,我们去吃烧烤店吃烤肉吧!",
                     "下班了，回家看书码字打游戏.",
                     "今天也是穷兮兮的一天呢,只有读书使我精神富裕.",
                     "将头发梳成大人模样,穿上一身帅气西装."]
    tfs_idf_model = SimilarityModel(document_list, mode="tf_idf")
    sentence = "饿了,想吃肉"
    sentence = Counter(jieba.cut(sentence))
    print(tfs_idf_model.get_max_score(sentence))
