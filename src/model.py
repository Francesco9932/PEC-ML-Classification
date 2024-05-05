import torch.nn as nn
from transformers import DistilBertModel, ElectraModel, BertModel

EMBEDDING_DIM = 768
HIDDEN_DIM_LSTM = 256


class BERTClassifier(nn.Module):
    def __init__(self, bert_finetuned_name, n_classes, base_model):
        super(BERTClassifier, self).__init__()

        if base_model == 'distilbert':
            self.bert = DistilBertModel.from_pretrained(
                bert_finetuned_name,  return_dict=False)
        elif base_model == 'electra':
            self.bert = ElectraModel.from_pretrained(
                bert_finetuned_name,  return_dict=False)
        elif base_model == 'bert':
            self.bert = BertModel.from_pretrained(
                bert_finetuned_name,  return_dict=False)
        print(f'Using {bert_finetuned_name}')
        self.drop = nn.Dropout(p=0.3)  # test 0.5 or 0.7
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state = self.bert(input_ids, attention_mask)
        output = self.drop(last_hidden_state[0][:, 0, :])
        return self.out(output)


class BERT_LSTMClassifier(nn.Module):
    def __init__(self, bert_finetuned_name, n_classes, base_model):
        super(BERTClassifier, self).__init__()

        if base_model == 'distilbert':
            self.bert = DistilBertModel.from_pretrained(
                bert_finetuned_name,  return_dict=False)
        elif base_model == 'electra':
            self.bert = ElectraModel.from_pretrained(
                bert_finetuned_name,  return_dict=False)
        elif base_model == 'bert':
            self.bert = BertModel.from_pretrained(
                bert_finetuned_name,  return_dict=False)
        print(f'Using {bert_finetuned_name}')
        self.lstm = nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM_LSTM, batch_first=True)
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(HIDDEN_DIM_LSTM, n_classes)

    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooled_output = self.bert(input_ids, attention_mask)
        lstm_output, (h_n, c_n) = self.lstm(last_hidden_state)
        h_n = h_n.view(-1, HIDDEN_DIM_LSTM)
        output = self.drop(h_n)
        return self.out(output)
