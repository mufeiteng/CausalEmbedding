###### 指标: auc
###### 结果
max-matching: 0.8
bmm: 0.87
* mm+knn: 0.87
* bmm+knn: 0.89

#### 模型尝试
* max-matching(concat(c,c'), concat(e, e'))不好,因为会涉及c'与e'的相似度计算
* c'=tahn(weighted_pooling)没有c'=weighted_pooling好
* pos_loss = reduce_sum(batch_loss*(2*pattern_prob-0.8))不好
* neg_loss = reduce_sum(loss)/number 不好

#### 参数尝试
* pa_prob_thres = 0.5 结果为0.88
* weak_pattern_prob -= 0.2 没啥变化
* auc(alpha1=0.5) > auc(alpha1=0.75)

#### sota
##### 损失为loss(c,e)+loss(c,e')+loss(c',e)
* c'=weighted_pooling(softmax) 0.88
* c'=weighted_pooling(divide_sum) 0.893

parameters = {
    'strong_w_count': 1,  # 强pattern的min-count
    'weak_w_count': 5,  # 弱pattern的min-count
    'pa_thres': 0.6,  # pattern的概率
    'weak_p_diff': False,
    'weak_p_offset': 0.2,
    'num_sample': 10,
    'max_context_len': 10,
    'max_len': 25,
    'trial_type': 6,
    'lr': 5e-3,
    'alpha1': 0.5,

}


#### bmm
1. auc: 0.893

'strong_w_count': 1,  # 强pattern的min-count
'weak_w_count': 5,  # 弱pattern的min-count
'pa_thres': 0.6,  # pattern的概率

'weak_p_diff': False,
'weak_p_offset': 0.2,
'max_context_len': 10,
'trial_type': 'sota',
'lr': 5e-3,
'alpha1': 0.5,

2. auc 0.886

'weak_p_diff': True,
'weak_p_offset': 0.1,
'max_context_len': 10,
'trial_type': 'sota',
'lr': 5e-3,
'alpha1': 0.7,

