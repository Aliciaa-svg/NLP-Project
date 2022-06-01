import json
import os
import numpy as np
import torch
import torch.utils.data as Data
from scorer import Scorer, Dataset
from transformers import *
from ModelUtils import CustomDataset, BertProcessor

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def read_jsonl_file(filename):
    return [json.loads(line) for line in open(filename, "r")]

def write_jsonl_file(data, filename):
    with open(filename, "w") as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    file = 'data/processed/train_scorer.jsonl'

    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
    processor = BertProcessor()
    train_dataset = CustomDataset(file, processor.read_file)

    model = Scorer(tokenizer)
    device = 'cuda'
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-4)
    epochs = 50
    criterion = torch.nn.CrossEntropyLoss()

    train_dataloader = processor.get_dataloader(
        dataset=train_dataset,
        args=None,
        tokenizer=tokenizer
    )
    max_acc = 0
    min_loss = 10000000
    num_steps = len(train_dataloader)
    print_per_step = max(10, int(num_steps / 10))
    for e in np.arange(epochs):
        # print(f'###### start epoch {e} ######')
        model.train()
        loss_epoch = 0
        acc_cnt = 0
        cnt = 0
        for step, x in enumerate(train_dataloader):
            
            optimizer.zero_grad()

            inputs, labels = x
            
            inputs["input_ids"] = inputs["input_ids"].cuda()
            inputs["attention_mask"] = inputs["attention_mask"].cuda()
            inputs["token_type_ids"] = inputs["token_type_ids"].cuda()
            
            labels = labels.cuda()
            
            outputs = model.model(**inputs)
            embed = outputs.pooler_output
            
            logit = model(embed)
            loss = criterion(logit, labels.long())

            acc_cnt += ((labels==logit.argmax(dim=1)).sum())
            cnt += len(labels)

            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

            if step % print_per_step == 0:
                print(
                    "Train:: EPOCH: {0}, Step: {1}/{2}, loss: {3}".format(
                        e, step, num_steps, round(loss.item(), 8)
                    )
                )
        
        acc = (acc_cnt / cnt).float()
        if acc > max_acc:
            max_acc = acc
            model_max_acc = model.state_dict()
        if loss_epoch < min_loss:
            min_loss = loss_epoch
            model_min_loss = model.state_dict()
        print(f"~~~~~~ Epoch {e} | Train Loss {loss_epoch} | Train Acc {acc} ~~~~~~")

    print(f"###### FINISH TRAINING ! Max Acc: {max_acc}, Min Loss: {min_loss} ######")
    torch.save(model_max_acc, f'model/scorer/max_acc.pth')
    torch.save(model_min_loss, f'model/scorer/min_loss.pth')
    torch.save(model.state_dict(), f'model/scorer/final.pth')