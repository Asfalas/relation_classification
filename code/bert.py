import torch, os
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertTokenizer
from transformers import get_linear_schedule_with_warmup, AdamW
from .data_utils import load_data_and_labels, label2class
from sklearn.model_selection import train_test_split
from .bert_preprocess import RCDataset
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.utils.data as Data


class BertForRelationClassification(nn.Module):
    def __init__(self):
        super(BertForRelationClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        # self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(768, 19)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0][:, 0, :]
        # sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


def train_bert(args):
    batch_size = args.train_batch_size
    maxlen = args.max_len
    epoches = args.num_train_epochs
    lr = args.learning_rate
    save_model_dir = args.save_model_path
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    add_CLS = True

    train_text, y, _ = load_data_and_labels('./data/TRAIN_FILE.TXT')
    train_data, dev_data, train_y, dev_y = train_test_split(train_text, y, test_size=0.1, random_state=2020)
    train_dataset = RCDataset(train_data, train_y, tokenizer, maxlen, add_CLS)
    dev_dataset = RCDataset(dev_data, dev_y, tokenizer, maxlen, add_CLS)

    train_data_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_data_loader = Data.DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)

    model = BertForRelationClassification()
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    loss_function = nn.CrossEntropyLoss()
    best_acc = 0
    early_stop = 0
    for epoch in range(epoches):
        print('l_r: ' + str(optimizer.state_dict()['param_groups'][0]['lr']))
        bar = tqdm(list(enumerate(train_data_loader)))
        for step, input_data in bar:
            input_ids = input_data[0].to(args.device)
            attention_mask = input_data[1].to(args.device)
            y1 = input_data[-1].to(args.device)
            model.train()
            model.zero_grad()
            pred = model(input_ids, attention_mask)
            loss = loss_function(pred, y1.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            prob1 = torch.log_softmax(pred, dim=1)
            pred_cls1 = torch.argmax(prob1, dim=1)
            acc1 = (pred_cls1.cpu() == y1.squeeze().cpu()).sum().numpy() / pred_cls1.size()[0]
            bar.set_description(f"step:{step}: acc:{acc1} loss:{loss}")
        scheduler.step()
        with torch.no_grad():
            model.eval()
            right_count = 0
            all_count = 0
            for step, test_input_data in tqdm(list(enumerate(dev_data_loader))):
                test_input_ids = test_input_data[0].to(args.device)
                test_attention_mask = test_input_data[1].to(args.device)
                test_y1 = test_input_data[-1].to(args.device)
                test_pred = model(test_input_ids, test_attention_mask)
                prob = torch.log_softmax(test_pred, dim=1)
                pred_cls = torch.argmax(prob, dim=1)
                right_count += (pred_cls.cpu() == test_y1.squeeze().cpu()).sum().numpy()
                all_count += pred_cls.cpu().size()[0]
            eval_acc = float(right_count) / float(all_count)
            print(f"---------------------eval:{epoch}: accuracy:{eval_acc}---------------------")
            if eval_acc > best_acc:
                torch.save(model.state_dict(), save_model_dir)
                test_bert(args, model)
                best_acc = eval_acc
                early_stop = 0
            else:
                early_stop += 1
        if early_stop >= 5:
            break
    return model


def test_bert(args, model):
    save_model_path = args.save_model_path
    out_path = args.output_path
    out_dir = args.output_dir
    maxlen = args.max_len
    batch_size = args.train_batch_size
    add_CLS = True

    x_text_dev, ground_truths, nums = load_data_and_labels('./data/TEST_FILE_FULL.TXT')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    test_dataset = RCDataset(x_text_dev, ground_truths, tokenizer, maxlen, add_CLS)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    total_pred_cls = []
    with torch.no_grad():
        model.eval()
        for step, input_data in tqdm(enumerate(test_dataloader)):
            test_input_ids = input_data[0].to(args.device)
            test_attention_mask = input_data[1].to(args.device)
            test_y1 = input_data[-1].to(args.device)

            out = model(test_input_ids, test_attention_mask)
            prob = torch.log_softmax(out, dim=1)
            pred_cls = torch.argmax(prob, dim=1)
            for cls in pred_cls:
                total_pred_cls.append(cls)

        f_out = open(out_path, 'w')
        for id, label in zip(nums, total_pred_cls):
            if int(label) == 0:
                f_out.write(str(id) + '\t' + label2class[int(label)] + '\n')
            else:
                f_out.write(str(id) + '\t' + label2class[int(label)] + '\n')
        f_out.close()

        f_keys = open(out_dir + 'keys.txt', 'w')
        for id, gt in zip(nums, ground_truths):
            if int(gt) == 0:
                f_keys.write(str(id) + '\t' + label2class[int(gt)] + '\n')
            else:
                f_keys.write(str(id) + '\t' + label2class[int(gt)] + '\n')
        f_keys.close()

        cmd = out_dir + "semeval2010_task8_scorer-v1.2.pl " + out_path + " " + out_dir + "keys.txt > " \
              + out_dir + "bert_result_scores.txt"
        print(cmd)
        os.system(cmd)
        with open(out_dir + "bert_result_scores.txt", 'r') as w:
            print(w.read())
