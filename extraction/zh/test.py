# 要高考了，我心乱如麻
# 因身材较矮，显得帽翅格外长
# 整个风景区庄严肃穆，环境优美；
# 神奇至极，令人不由慨叹大自然的鬼斧神工
# 牛郎无望，只能停止追赶
# 天山英雄传的动作很新颖，不单调、枯燥
# 解脱之法用尽，终郁郁不得，故长叹与君知
# 君令人之侧目，在于文笔生花，话语如珠
# 山茶油能改变食用单一油类所造成的营养不均，能较充分地平衡人体营养，有利身体健康
# ptree = PTree(LTPModel())
# s = '作为一款全球首发的车型，新天籁在北京车展的亮相还是引起了不少观众的关注。'
# ptree.create(s)
# result = ptree.get_causal_result('verb')
# print(result)
# utils = Utils()
# utils.reset(ptree=ptree, result=result)
# response = Cause_V_Effect.boost(utils)
# print(response)

from msworks.extraction.cn.parser import CnParseTree, LTPModel
from extraction.zh.causalrules import *
ltp_model = LTPModel()
ptree = CnParseTree(ltp_model)

# print(158*768)


# def process_a_line(line):
#     sents = ltp_model.sentencesplitter.split(line)
#     lines = []
#     for sent in sents:
#         if sent.strip():
#             try:
#                 sent = re.sub('[^，。；:、（）《》“”0-9\u4e00-\u9fa5]', '', sent)
#                 sent = sent.strip()
#                 if len(sent) < 7 or len(sent) > 200:
#                     continue
#                 lines.append(sent)
#             except Exception as e:
#                 print(e)
#     return lines
#
# count = 0
# path = '/home/feiteng/Documents/sources/sg_data/raw/news.sohunews.1260804.txt'
# # fin1 = codecs.open(path+'temp.txt', 'r', 'utf-8')
# fin2 = codecs.open(path, 'r', 'utf-8')
# line = fin2.readline()
# while line:
#     sents = process_a_line(line)
#     for s in sents:
#         ptree.create(s)
#         response = CandidateRules.boost(ptree)
#         if response:
#             for res in response:
#                 print(s)
#                 print(res['cue'], res['left'], res['right'])
#                 print()
#                 count += 1
#     line = fin2.readline()
# print(count)

# 与水性好的沐童是死对头，两人在日后的生活中，麻烦不断，使他们的矛盾升级----麻烦(a) 断(v)----矛盾(a) 升级(v)
# 连绵的棘林偶尔会间有小片的棕榈林、盐土乾草原和由火或砍伐造成的稀树草原
# 用途用于各种功能性心律失常、室上性及室性异位期外收缩、心房纤维颤动和麻醉引起的心律不齐等
# 煎蛋炸鸡容易引发妇科恶疾----煎蛋(v) 炸鸡(n) 容易(a)----妇科恶疾(n)
# 元凶2∶过度劳累引发的腿抽筋
# 近年试作红日、雪林等，具现代气息，使人耳目一新----具(v) 气息(n)----人(n) 耳目一新(i)
# 血腥可造成的物理伤害增加15百分号持续15秒消耗35 //修饰关键词的副词 舍弃

# 自然灾害所造成的食物严重缺乏使某动物种群大量饥饿致死
# 1000年前，“播种者”在火星上培育的生物由于彗星的撞击导致毁灭----播种者(n) 火星(n) 培育(v) 生物(n)----毁灭(v)
# 怀孕期间胎儿父亲因他人侵权行为造成死亡的，婴儿出生后享有请求赔偿的权利王德钦诉杨德胜、泸州市汽车二队交通事故损害赔偿纠纷案----胎儿(n) 父亲(n)----死亡(v)
# 笔记本的丢失同样也可能引起数据丢失 //两个VOB

s = '与水性好的沐童是死对头，两人在日后的生活中，麻烦不断，使他们的矛盾升级'
ptree.create(s)
res = PositivesRules.extract(ptree)
print(res)
