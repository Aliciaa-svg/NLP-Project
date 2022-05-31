"""
Preprocess Data For Training
"""
import json
from text2vec import SentenceModel, EncoderType, cos_sim
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def get_sim_score(text1, text2, encoder):
    embed1 = encoder.encode(text1)
    embed2 = encoder.encode(text2)
    score = float(cos_sim(embed1, embed2)[0])
    return score

def read_jsonl_file(filename):
    return [json.loads(line) for line in open(filename, "r")]

def write_jsonl_file(data, filename):
    with open(filename, "w") as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + "\n")

def generate_data_for_model_userbase_tri(ori_data):
    """
    [NOTE] 
     -- generate "service-user" pairs & "user-service" pairs & "service-user-service" pairs #{more \times 2} flexible
     -- save CANDIDATES for training scorer
    """
    # ori_data = read_jsonl_file(filename)
    data4model = []
    encoder = SentenceModel("shibing624/text2vec-base-chinese",
                              encoder_type=EncoderType.FIRST_LAST_AVG)
    role_dict = {
        'service': '销售',
        'user': '客户'
    }
    scorer_data = [] 
    for index, one_dialogue in enumerate(ori_data):
        print("process data {}".format(index))
        dialogues, summary = one_dialogue["dialogues"], one_dialogue["summary"]
        candidates = []
        candidates_norole = []
        
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
                candidates.append(tmp)
                candidates_norole.append(tmp_norole)

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
                candidates.append(tmp)
                candidates_norole.append(tmp_norole)
        
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
                candidates.append(tmp_reverse)
                candidates_norole.append(tmp_norole)

        flag = np.zeros(len(candidates))
        for summ in summary:
            max_lcs = 0
            fit_dialogue = None
            select_idx = None
            for idx, candi in enumerate(candidates):
                lcs = get_sim_score(summ, "".join(candi), encoder)
                if lcs > max_lcs:
                    max_lcs = lcs
                    fit_dialogue = candi.copy()
                    select_idx = idx
            if fit_dialogue:
                src_txt = "#".join(fit_dialogue)
                data4model.append(
                    {
                        "src_txt": src_txt,
                        "tgt_txt": summ,
                    }
                )
                flag[select_idx] = 1

        for f, candi in enumerate(candidates):
            src = "#".join(candi)
            if flag[f] == 1:
                scorer_data.append({
                    "src_txt": src,
                    "class": 1
                })
            else:
                scorer_data.append({
                    "src_txt": src,
                    "class": 0
                })
    
    return data4model, scorer_data

if __name__ == "__main__":
    train_data = read_jsonl_file('data/train.jsonl')
    data_abs, data_score = generate_data_for_model_userbase_tri(train_data)

    write_jsonl_file(data_abs, "data/processed/train_abstract.jsonl")
    write_jsonl_file(data_score, "data/processed/train_scorer.jsonl")
