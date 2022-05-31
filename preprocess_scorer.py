import json
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def read_jsonl_file(filename):
    return [json.loads(line) for line in open(filename, "r")]


def write_jsonl_file(data, filename):
    with open(filename, "w") as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + "\n")

def generate_data_for_model_userbase_tri(ori_data, select_data):
    """
    [NOTE] 
     -- generate "service-user" pairs & "user-service" pairs & "service-user-service" pairs #{more \times 2} flexible
     -- save CANDIDATES for training scorer
    """
    data4model = []
    role_dict = {
        'service': '销售',
        'user': '客户'
    }
    candidates_all = []
    for index, one_dialogue in enumerate(ori_data):
        print("process data {}".format(index))
        dialogues, summary = one_dialogue["dialogues"], one_dialogue["summary"]
        
        # combine near 5 utterances starting wiz "user"
        for i, d in enumerate(dialogues):
            role = d["role_type"]
            text = d["text"]
            
            if role == "user":
                tmp = []
                tmp_norole = []
                j = i-2
                while (j-i) <= 2:
                    if j < 0:
                        j += 1
                        continue
                    if j == len(dialogues): break
                    d_next = dialogues[j]
                    r_next = role_dict[d_next["role_type"]]
                    t_next = d_next["text"]
                    tmp.append(r_next + "：" + t_next)
                    tmp_norole.append(t_next)
                    j += 1
                candidates_all.append(tmp)
        
        for i, d in enumerate(dialogues):
            role = d["role_type"]
            text = d["text"]
            
            if role == "user":
                tmp = []
                tmp_norole = []
                tmp.append(role_dict[role]+"："+text)
                tmp_norole.append(text)
                j = i+1
                while (j-i) <= 4:
                    if j == len(dialogues): break
                    d_next = dialogues[j]
                    r_next = role_dict[d_next["role_type"]]
                    t_next = d_next["text"]
                    tmp.append(r_next + "：" + t_next)
                    tmp_norole.append(t_next)
                    j += 1
                # tmp_c = ' '.join(tmp)
                candidates_all.append(tmp)
                
        for i in range(len(dialogues)-1, -1, -1):
            d = dialogues[i]
            role = d["role_type"]
            text = d["text"]
            
            if role == "user":
                tmp = []
                tmp_norole = []
                tmp.append(role_dict[role]+"："+text)
                tmp_norole.append(text)
                j = i-1
                while (i-j) <= 4:
                    if j == -1: break
                    d_next = dialogues[j]
                    r_next = role_dict[d_next["role_type"]]
                    t_next = d_next["text"]
                    tmp.append(r_next + "：" + t_next)
                    tmp_norole.append(t_next)
                    j -= 1
                tmp_reverse = []
                for j in range(len(tmp)-1, -1, -1):
                    tmp_reverse.append(tmp[j])
                # tmp_c = ' '.join(tmp)
                candidates_all.append(tmp_reverse)

    scorer_data = {}
    for c in candidates_all:
        content = '#'.join(c)
        scorer_data[content] = 0
    for s in select_data:
        content = s['src_txt']
        scorer_data[content] = 1
    
    return data4model, scorer_data

if __name__ == "__main__":
    
    ori_data = read_jsonl_file("data/train.jsonl")
    select_data = read_jsonl_file("data/processed/train_abstract.jsonl")

    data_abs, data_score = generate_data_for_model_userbase_tri(ori_data, select_data)

    cnt = 0
    ll = list(data_score.keys())
    data = []
    for idx, l in enumerate(ll):
        # print(f'processing data {idx}')
        label = data_score[l]
        if label == 1:
            data.append({
                'src_txt': l,
                'class': label
            })
        else:
            cnt += 1
            if cnt >= 10000:
                continue
            else:
                data.append({
                'src_txt': l,
                'class': label
                })

    write_jsonl_file(data, "data/train_scorer.jsonl")
    