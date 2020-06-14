import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from tqdm import tqdm
from .data_utils import load_data_and_labels, load_glove, label2class
import torch.utils.data as Data
from .vocabulary import Vocab
import os, pickle

torch.manual_seed(2020)


class AttBiLSTM(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_size,
                 hidden_size, pretrained_weight, keep_prob=0.5):
        super(AttBiLSTM, self).__init__()
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob

        # embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim=embedding_size)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding_layer.weight.requires_grad = True
        self.dropout_embed = nn.Dropout(self.keep_prob)

        # Bi_LSTM layer
        self.bi_lstm_layer = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size // 2, num_layers=2,
                                     bidirectional=True,
                                     batch_first=True, dropout=self.keep_prob)

        # attention layer
        self.attention_layer = SelfAttention(hidden_size)

        # FC layer
        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.num_classes)
        )

    def forward(self, inputs):
        # inputs = (B, S, W)
        # embed = self.embedding_layer(inputs)
        embed = self.dropout_embed(self.embedding_layer(inputs))
        # embed = (B, S, E)
        o, (h, c) = self.bi_lstm_layer(embed)
        # lstm_out = o[:, :, :self.hidden_size] + o[:, :, self.hidden_size:]
        lstm_out = o
        # lstm_out = (B, S, H)
        att_out, weights = self.attention_layer(lstm_out)
        # att_out = (B, 0, H)
        out = self.fc_layer(att_out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = torch.nn.functional.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


def train_att_bilstm(args):
    embedding_size = args.embedding_dim
    hidden_size = args.hidden_dim
    batch_size = args.train_batch_size
    maxlen = args.max_len
    epochs = args.num_train_epochs
    lr = args.learning_rate
    save_model_dir = args.save_model_path
    num_labels = args.num_labels
    keep_prob = args.keep_prob

    train_text, y, _ = load_data_and_labels('./data/TRAIN_FILE.TXT')
    vocab = Vocab(train_text, maxlen)
    x = vocab.sentence_list_to_np(train_text)
    print("Text Vocabulary Size: {:d}".format(len(vocab.word_to_idx_dict)))
    print("x = {0}".format(x.shape))
    print("y = {0}".format(y.shape))

    # Randomly shuffle data to split into train and test(dev)
    np.random.seed(2020)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(0.1 * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    glove = load_glove('code/vector_cache/glove.6B.' + str(embedding_size) + 'd.txt', embedding_size, vocab)
    model = AttBiLSTM(num_classes=num_labels, vocab_size=len(vocab.word_to_idx_dict), embedding_size=embedding_size,
                      hidden_size=hidden_size, pretrained_weight=glove, keep_prob=keep_prob)
    model.to(args.device)
    train_data = Data.TensorDataset(torch.tensor(x_train, dtype=torch.int64), torch.tensor(y_train, dtype=torch.int64))

    test_x = torch.tensor(x_dev, dtype=torch.int64)
    test_y = torch.tensor(y_dev, dtype=torch.int64)

    train_data_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    best_acc = 0
    early_stop = 0
    for epoch in range(epochs):
        print('l_r: ' + str(optimizer.state_dict()['param_groups'][0]['lr']))
        bar = tqdm(list(enumerate(train_data_loader)))
        for step, input_data in bar:
            x1, y1 = input_data
            model.train()
            pred = model(x1.to(args.device))
            loss = loss_function(pred, y1.squeeze().to(args.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prob1 = torch.log_softmax(pred, dim=1)
            pred_cls1 = torch.argmax(prob1, dim=1)
            acc1 = (pred_cls1.cpu() == y1.squeeze().cpu()).sum().numpy() / pred_cls1.cpu().size()[0]
            bar.set_description(f"step:{step}: acc:{acc1} loss:{loss}")
        scheduler.step()
        with torch.no_grad():
            model.eval()
            test_pred = model(test_x.to(args.device))
            prob = torch.log_softmax(test_pred, dim=1)
            pred_cls = torch.argmax(prob, dim=1)
            eval_acc = (pred_cls.cpu() == test_y.cpu().squeeze()).sum().numpy() / pred_cls.cpu().size()[0]
            print(f"---------------------eval:{epoch}: accuracy:{eval_acc}---------------------")
            if eval_acc > best_acc:
                torch.save(model.state_dict(), save_model_dir)
                best_acc = eval_acc
                early_stop = 0
            else:
                early_stop += 1
        if early_stop >= 5:
            break
    return model


def test_att_bilstm(args):
    embedding_size = args.embedding_dim
    hidden_size = args.hidden_dim
    num_labels = args.num_labels
    keep_prob = args.keep_prob

    save_model_path = args.save_model_path
    out_path = args.output_path
    out_dir = args.output_dir
    # 加载字典
    train_text, y, _ = load_data_and_labels('./data/TRAIN_FILE.TXT')
    vocab = Vocab(train_text, args.max_len)
    glove = load_glove('code/vector_cache/glove.6B.' + str(embedding_size) + 'd.txt', embedding_size, vocab)
    model = AttBiLSTM(num_classes=num_labels, vocab_size=len(vocab.word_to_idx_dict), embedding_size=embedding_size,
                      hidden_size=hidden_size, pretrained_weight=glove, keep_prob=keep_prob)

    model.load_state_dict(torch.load(save_model_path))
    model.to(args.device)
    model.eval()

    x_text_dev, ground_truths, nums = load_data_and_labels('./data/TEST_FILE_FULL.TXT')
    x_dev = list(vocab.sentence_list_to_np(x_text_dev))

    out = model(torch.tensor(x_dev, dtype=torch.int64).to(args.device))
    prob = torch.log_softmax(out, dim=1)
    pred_cls = torch.argmax(prob, dim=1)
    print(pred_cls)
    f_out = open(out_path, 'w')
    for id, label in zip(nums, pred_cls):
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

    cmd = out_dir + "/semeval2010_task8_scorer-v1.2.pl " + out_path + " " + out_dir + "keys.txt > " \
          + out_dir + "att_bilstm_result_scores.txt"
    print(cmd)
    os.system(cmd)
    with open(out_dir + "att_bilstm_result_scores.txt", 'r') as w:
        print(w.read())
