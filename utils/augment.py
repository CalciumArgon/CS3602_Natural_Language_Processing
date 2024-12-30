import json, random, copy
random.seed(0)

poi_slots = ['poi名称', 'poi修饰', 'poi目标', '起点名称', '起点修饰', '起点目标', 
                '终点名称', '终点修饰', '终点目标', '途经点名称']
poi_values = [c.rstrip() for c in open('data/lexicon/poi_name.txt',encoding="utf-8")]

poi_act = [c.rstrip() for c in open('data/lexicon/operation_verb.txt',encoding="utf-8")]
poi_number = [c.rstrip() for c in open('data/lexicon/ordinal_number.txt',encoding="utf-8")]

## 请求类型
# request_values = ["附近", "定位", "近郊", "旁边", "周边", "就近", "最近"]
request_values = [
    "附近", "近处", "靠近", "接近", "相邻", "周围",
    "定位", "位置", "所在地", "确定位置",
    "近郊", "郊区", "周边地区", "市郊",
    "旁边", "侧面", "旁边的", "隔壁",
    "周边", "附近", "邻近", "周围", "附近地区",
    "就近", "附近", "靠近", "便近",
    "最近", "近期", "近来", "最新", "近日"
]

#  路线偏好
# preferenece_values = ["最近", "高速优先", "走国道", "少走高速", "不走高速", "走高速",
#             "上高速", "高速公路", "最快", "躲避拥堵"]
preferenece_values = [
    "最近", "高速优先", "走国道", "少走高速", "不走高速", "走高速",
    "上高速", "高速公路", "最快", "躲避拥堵",
    "近期", "优先考虑高速", "选择国道路线", "避免高速", "避开高速", "选择高速",
    "进入高速", "高速公路", "最迅速", "避免交通拥堵"
]
# 对象
# object_values = ["语音", "高德地图", "路线", "位置", "途经点", "全程路线",
#           "简易导航", "目的地", "地图", "定位", "路况", "导航"]
object_values = [
    "语音", "高德地图", "路线", "位置", "途经点", "全程路线",
    "简易导航", "目的地", "地图", "定位", "路况", "导航",
    "语音导航", "地图导航", "导航路线", "当前位置", "导航目的地", "导航地图",
    "实时定位", "实时路况", "导航系统"]

def AugSemantics(utt):
    new_utt = copy.deepcopy(utt)
    for pair_3 in new_utt["semantic"]:
        if pair_3[1] in poi_slots:
            new_value = random.choice(poi_values)
            new_utt["asr_1best"] = new_utt["asr_1best"].replace(pair_3[2], new_value)
            pair_3[2] = new_value

        if True: # Ture -- only aug poi; 
            continue
        elif pair_3[1] == "操作":
            new_value = random.choice(poi_act)
            new_utt["asr_1best"] = new_utt["asr_1best"].replace(pair_3[2], new_value)
            pair_3[2] = new_value
        elif pair_3[1] == "序列号":
            new_value = random.choice(poi_number)
            new_utt["asr_1best"] = new_utt["asr_1best"].replace(pair_3[2], new_value)
            pair_3[2] = new_value
        elif pair_3[1] == "请求类型":
            new_value = random.choice(request_values)
            new_utt["asr_1best"] = new_utt["asr_1best"].replace(pair_3[2], new_value)
            pair_3[2] = new_value
        elif pair_3[1] == "路线偏好":
            new_value = random.choice(preferenece_values)
            new_utt["asr_1best"] = new_utt["asr_1best"].replace(pair_3[2], new_value)
            pair_3[2] = new_value
        elif pair_3[1] == "对象":
            new_value = random.choice(object_values)
            new_utt["asr_1best"] = new_utt["asr_1best"].replace(pair_3[2], new_value)
            pair_3[2] = new_value
    new_utt["manual_transcript"] = new_utt["asr_1best"]
    return new_utt


train_data = json.load(open("data/train.json", 'r', encoding='utf-8'))

expansion_rate = 5
aug_data = []
for di, dialog in enumerate(train_data):
    aug_data.append(dialog)
    for i in range(expansion_rate): 
        aug_dialog = []
        for ui, utt in enumerate(dialog):
            aug_utt = AugSemantics(utt)
            aug_dialog.append(aug_utt)
        aug_data.append(aug_dialog)

aug_data_path = "data/train_aug.json"
json.dump(aug_data, open(aug_data_path, 'w',encoding='utf-8'), indent=4, ensure_ascii=False)