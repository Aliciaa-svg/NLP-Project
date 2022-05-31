import json
import os
from scorer import * 

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def read_jsonl_file(filename):
    return [json.loads(line) for line in open(filename, "r")]


def write_jsonl_file(data, filename):
    with open(filename, "w") as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + "\n")


def generate_test_data_for_model_with_scorer(filename):
    """
    [NOTE] preprocessing test data, add scorer
    """
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    scorer = Scorer(tokenizer)
    scorer = scorer.cuda()
    dict = torch.load('model/scorer/min_loss.pth')
    scorer.load_state_dict(dict)
    role_dict = {
        'service': '销售',
        'user': '客户'
    }
    ori_data = read_jsonl_file(filename)
    data4model = []
    
    for index, one_dialogue in enumerate(ori_data):
        print("process data {}".format(index))
        dialogues, summary = one_dialogue["dialogues"], one_dialogue["summary"]
        candidates = []

        i = 0
        while i < len(dialogues):
            d = dialogues[i]
            role = d["role_type"]
            text = d["text"]
            
            if role == "user":
                flag = None
                proposals = []
                """case 1: `user-service`"""
                j = i+1
                tmp = []
                tmp.append(role_dict[role]+"："+text)
                while (j-i) <= 3:
                    if j == len(dialogues): break
                    d_next = dialogues[j]
                    r_next = role_dict[d_next["role_type"]]
                    t_next = d_next["text"]
                    tmp.append(r_next + "：" + t_next)
                    j += 1
                proposals.append('#'.join(tmp))
                
                """case 2: `service-user`"""
                j = i - 3
                tmp = []
                while (j-i) <= 0:
                    if j < 0: 
                        j += 1
                        continue
                    d_next = dialogues[j]
                    r_next = role_dict[d_next["role_type"]]
                    t_next = d_next["text"]
                    tmp.append(r_next + "：" + t_next)
                    j += 1
                proposals.append('#'.join(tmp))

                """case 3: `service-user-service`"""
                j = i - 2
                tmp = []
                while (j-i) <= 2:
                    if j < 0:
                        j += 1
                        continue
                    if j == len(dialogues): break
                    d_next = dialogues[j]
                    r_next = role_dict[d_next["role_type"]]
                    t_next = d_next["text"]
                    tmp.append(r_next + "：" + t_next)
                    j += 1
                i += 2
                proposals.append('#'.join(tmp))

                """score proposals"""
                inputs = tokenizer(
                    proposals,
                    return_tensors = 'pt',
                    max_length = 320,
                    truncation = True,
                    padding = True
                )
                inputs["input_ids"] = inputs["input_ids"].cuda()
                inputs["attention_mask"] = inputs["attention_mask"].cuda()
                inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
                with torch.no_grad():
                    outputs = scorer.model(**inputs)
                    embed = outputs.pooler_output
                    scores = scorer(embed)[:, 1].cpu()
                    idx = scores.argmax()
                select_p = proposals[idx]
                candidates.append(select_p)
            i += 1

        for c in candidates:
            data4model.append(
                {
                    "src_txt": c,
                    "tgt_txt": '',
                    "id": index,
                }
            )
        
    return data4model

if __name__ == "__main__":
    
    data = generate_test_data_for_model_with_scorer(
        "data/test.jsonl")
    write_jsonl_file(data, "data/proposal.jsonl")
