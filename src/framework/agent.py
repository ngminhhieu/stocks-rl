import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, input_shape, embedding_dim, hidden_dim):
        super(Agent, self).__init__()
        self.embed = nn.Linear(input_shape, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, state):
        embed_o = self.embed(state)
        lstm_o = self.lstm(embed_o)[0]
        lstm_o = lstm_o[:, -1, :]
        dropout_o = self.dropout(lstm_o)
        relu_o = self.relu(dropout_o)
        fc_o = self.fc(relu_o)
        probs = self.softmax(fc_o)
        return probs