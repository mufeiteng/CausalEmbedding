## 讨论 12/16

### 实验

##### 数据

* bk_verb_positives.txt 20w个高精度的短语对

* bk_negatives.txt 抽取90w个其他关系的短语对

**理想结果：bk_verb_positives.txt中的高精度动词得到较高权重**

##### compare1

使用上面数据重新训练max-matching
结果：total 311, acc 0.3215434083601286, mrr 0.5015585077099166.

##### compare2

样本的label有causal embed给出，根据此label重新训练embedding。损失为向量计算得到的交叉熵
结果：total 308, acc 0.2792207792207792, mrr 0.4624904404666158.

##### compare3

在compare2的基础上加上pattern的权重,计算三个损失

加上了pattern的频次，对pattern频次求 softmax，与pattern权重相乘，计算sigmoid损失

**结果：前几个epoch，用pattern评估结果比embedding评估结果要高**，但是训练慢，收敛慢

### 可能问题

* 计算词对pairwise的概率的最大值

| 概率区间(<=) | bk_verb_positives.txt | bk_negatives.txt |
| ------------ | --------------------- | ---------------- |
| 0.1          | 4540                  | 14051            |
| 0.2          | 12011                 | 68041            |
| 0.3          | 26147                 | 193508           |
| 0.4          | 28529                 | 150877           |
| 0.5          | 14738                 | 26077            |
| 0.6          | 7159                  | 8005             |
| 0.7          | 1838                  | 1548             |
| 0.8          | 258                   | 213              |
| 0.9          | 13                    | 9                |
| 1.0          | 0                     | 0                |

**可以看出：**

* bk_verb_positives.txt中， $number_{p>=0.5}:number_{p<0.5} \approx 1:9$

* bk_negatives.txt中， $number_{p>=0.5}:number_{p<0.5} \approx 1:50$

所以即使在高质量预料中，也会有绝大多数样本给予0标签

##### 改进label采样

- 设阈值打标签，每个概率给固定的label，不通过采样给出
- 加入词频过滤，考虑embed打分，考虑排序信息
- 中间的不用来计算
- 本质上跟seed选取是一样的，因为是通过词对打分得出label

#### 提升泛化能力(3种)

* 词向量扩展：使用vanilla embed的相似性(上下位词，同义词，近义词替换)进行扩展。
* 知识库扩展：利用弱因果动词，得到大量语料，提升coverage
* pattern扩展：通过词对建模pattern，利用pattern的迁移能力，在给定文档中寻找因果关系

#### 滚数据

* 给定初始seed，滚pattern，再滚三元组
* 枚举所有的动词，抽取短语对，训练向量(类似word2vec，不区分原因结果)

最终得到<短语对，pattern>的数据，使用者部分数据进行训练

#### seed质量

* 过滤词对的词频
* 考虑排序信息
* 加入embedding计算得分



寻找一个感兴趣的方向一直做下去
跟风,下手要快
多读文章
自己做的事情跟什么有关系,进而去了解(远监督,半监督,bootstrap)

英文跑起来