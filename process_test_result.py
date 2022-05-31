import json

def read_jsonl_file(filename):
    return [json.loads(line) for line in open(filename, "r")]

def write_jsonl_file(data, filename):
    with open(filename, "w") as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    src = 'data/proposal.jsonl'
    tgt = 'data/result/result.txt'

    src_data = read_jsonl_file(src)

    with open(tgt, 'r') as t:
        tgt_data = t.readlines()
    summ = {}
    for i in range(100):
        summ[i] = []

    for s, t in zip(src_data, tgt_data):
        id = s['id']
        
        t = t.strip('\n')
        summ[id].append(t)

    f_write = 'data/test.jsonl'
    data_todo = read_jsonl_file(f_write)

    for i in range(len(data_todo)):
       data_todo[i]['summary'] = summ[i]
    
    f = 'data/result/result.jsonl'
    write_jsonl_file(data_todo, f)

