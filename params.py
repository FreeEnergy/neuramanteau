class Hyperparams(object):
    def __init__(self, batch_size, num_epochs, learn_rate, num_experts, embedding_sizes, hidden_sizes, dropout_probs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learn_rate = learn_rate
        self.num_experts = num_experts
        
        assert isinstance(hidden_sizes, list), 'hidden_sizes must be a list'
        assert isinstance(embedding_sizes, list), 'embedding_sizes must be a list'
        assert isinstance(dropout_probs, list), 'dropout_probs must be a list'
        self.hidden_sizes = hidden_sizes
        self.embedding_sizes = embedding_sizes
        self.dropout_probs = dropout_probs

    def __str__(self):
        return (
            'batch_size=' + str(self.batch_size) + '\n' +
            'num_epochs=' + str(self.num_epochs) + '\n' +
            'learn_rate=' + str(self.learn_rate) + '\n' +
            'num_experts=' + str(self.num_experts) + '\n' + 
            'hidden_sizes=' + str(self.hidden_sizes) + '\n' +
            'embedding_sizes=' + str(self.embedding_sizes) + '\n' +
            'dropout_probs=' + str(self.dropout_probs)
        )
