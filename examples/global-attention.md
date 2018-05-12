
# Sequence to sequence model with global attention

Paper: [Effective Approaches to Attention-based Neural Machine Translation](http://aclweb.org/anthology/D15-1166)



## Neural-Machine-Translation

Code: [mohamedkeid/Neural-Machine-Translation](https://github.com/mohamedkeid/Neural-Machine-Translation)

PyTorch implementation of "Effective Approaches to Attention-based Neural Machine Translation" using [scheduled sampling](https://arxiv.org/pdf/1506.03099.pdf) to improve the parameter estimation process. 

![](../images/cover.jpg)


### EncoderRNN

```python
class EncoderRNN(nn.Module):
    """Recurrent neural network that encodes a given input sequence."""

    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        seq_len = len(word_inputs)
        embedded = self.embedding(word_inputs).view(seq_len, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def init_hidden(self):
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        hidden = hidden.cuda()
        return hidden
```

### Attention

```python
class Attention(nn.Module):
    """Attention nn module that is responsible for computing the alignment scores."""

    def __init__(self, method, hidden_size):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        #Define layers
        if self.method == 'general':
            self.attention = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attention = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden, encoder_outputs):
        """Attend all encoder inputs conditioned on the previous hidden state of the decoder.
        
        After creating variables to store the attention energies, calculate their 
        values for each encoder output and return the normalized values.
        
        Args:
            hidden: decoder hidden output used for condition
            encoder_outputs: list of encoder outputs
            
        Returns:
             Normalized (0..1) energy values, re-sized to 1 x 1 x seq_len
        """

        seq_len = len(encoder_outputs)
        energies = Variable(torch.zeros(seq_len)).cuda()
        for i in range(seq_len):
            energies[i] = self._score(hidden, encoder_outputs[i])
        return F.softmax(energies).unsqueeze(0).unsqueeze(0)

    def _score(self, hidden, encoder_output):
        """Calculate the relevance of a particular encoder output in respect to the decoder hidden."""

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
        elif self.method == 'general':
            energy = self.attention(encoder_output)
            energy = hidden.dot(energy)
        elif self.method == 'concat':
            energy = self.attention(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dor(energy)
        return energy
```

### AttentionDecoderRNN

```python
class AttentionDecoderRNN(nn.Module):
    """Recurrent neural network that makes use of gated recurrent units to translate encoded inputs using attention."""

    def __init__(self, attention_model, hidden_size, output_size, n_layers=1, dropout_p=.1):
        super(AttentionDecoderRNN, self).__init__()
        self.attention_model = attention_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        #Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)

        #Choose attention model
        if attention_model is not None:
            self.attention = Attention(attention_model, hidden_size)

    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        """Run forward propagation one step at a time.
        
        Get the embedding of the current input word (last output word) [s = 1 x batch_size x seq_len]
        then combine them with the previous context. Use this as input and run through the RNN. Next,
        calculate the attention from the current RNN state and all encoder outputs. The final output
        is the next word prediction using the RNN hidden state and context vector.
        
        Args:
            word_input: torch Variable representing the word input constituent
            last_context: torch Variable representing the previous context
            last_hidden: torch Variable representing the previous hidden state output
            encoder_outputs: torch Variable containing the encoder output values
            
        Return:
            output: torch Variable representing the predicted word constituent 
            context: torch Variable representing the context value
            hidden: torch Variable representing the hidden state of the RNN
            attention_weights: torch Variable retrieved from the attention model
        """

        #Run through RNN
        word_embedded = self.embedding(word_input).view(1, 1, -1)
        rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_output, hidden = self.gru(rnn_input, last_hidden)

        #Calculate attention
        attention_weights = self.attention(rnn_output.squeeze(0), encoder_outputs)
        context = attention_weights.bmm(encoder_outputs.transpose(0, 1))

        #Predict output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)))
        return output, context, hidden, attention_weights
```

### train

```python
def train(input_var, target_var, encoder, decoder, encoder_opt, decoder_opt, criterion):
    #Initialize optimizers and loss
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()
    loss = 0

    #Get input and target seq lengths
    target_length = target_var.size()[0]

    #Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_var, encoder_hidden)

    #Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([0]))
    decoder_input = decoder_input.cuda()
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_context = decoder_context.cuda()
    decoder_hidden = encoder_hidden

    #Scheduled sampling
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        #Feed target as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                         decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output[0], target_var[di])
            decoder_input = target_var[di]
    else:
        #Use previous prediction as next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input,
                                                                                         decoder_context,
                                                                                         decoder_hidden,
                                                                                         encoder_outputs)
            loss += criterion(decoder_output[0], target_var[di])

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda()

            if ni == 1:
                break

    #Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
    encoder_opt.step()
    decoder_opt.step()

    return loss.data[0] / target_length
```



## seq2seq-pytorch


Code: [alanwang93/seq2seq-pytorch](https://github.com/alanwang93/seq2seq-pytorch)


Seq2seq model with

* global attention
* self-critical sequence training


### EncoderRNN

```python
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, padding_idx=1):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers)

    def forward(self, inputs, lengths, return_packed=False):
        """
        Inputs:
            inputs: (seq_length, batch_size), non-packed inputs
            lengths: (batch_size)
        """
        #[seq_length, batch_size, embed_length]
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, lengths=lengths.numpy())
        outputs, hiddens = self.rnn(packed)
        if not return_packed:
            return pad_packed_sequence(outputs)[0], hiddens
        return outputs, hiddens
```

### GlobalAttention

```python
class GlobalAttention(nn.Module):
    """
    Global Attention as described in 'Effective Approaches to Attention-based Neural Machine Translation'
    """
    def __init__(self, enc_hidden, dec_hidden):
        super(GlobalAttention, self).__init__()
        self.enc_hidden = enc_hidden
        self.dec_hidden = dec_hidden

        #a = h_t^T W h_s
        self.linear_in = nn.Linear(enc_hidden, dec_hidden, bias=False)
        #W [c, h_t]
        self.linear_out = nn.Linear(dec_hidden + enc_hidden, dec_hidden)
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()

    def sequence_mask(self, lengths, max_len=None):
        """
        Creates a boolean mask from sequence lengths.
        """
        batch_size = lengths.numel()
        max_len = max_len or lengths.max()
        return (torch.arange(0, max_len)
                .type_as(lengths)
                .repeat(batch_size, 1)
                .lt(lengths.unsqueeze(1)))

    def forward(self, inputs, context, context_lengths):
        """
        input (FloatTensor): batch x tgt_len x dim: decoder's rnn's output. (h_t)
        context (FloatTensor): batch x src_len x dim: src hidden states
        context_lengths (LongTensor): the source context lengths.
        """
        #(batch, tgt_len, src_len)
        align = self.score(inputs, context)
        batch, tgt_len, src_len = align.size()


        mask = self.sequence_mask(context_lengths)
        #(batch, 1, src_len)
        mask = mask.unsqueeze(1)  # Make it broadcastable.
        if next(self.parameters()).is_cuda:
            mask = mask.cuda()
        align.data.masked_fill_(1 - mask, -float('inf')) # fill <pad> with -inf

        align_vectors = self.softmax(align.view(batch*tgt_len, src_len))
        align_vectors = align_vectors.view(batch, tgt_len, src_len)

        #(batch, tgt_len, src_len) * (batch, src_len, enc_hidden) -> (batch, tgt_len, enc_hidden)
        c = torch.bmm(align_vectors, context)

        #\hat{h_t} = tanh(W [c_t, h_t])
        concat_c = torch.cat([c, inputs], 2).view(batch*tgt_len, self.enc_hidden + self.dec_hidden)
        attn_h = self.tanh(self.linear_out(concat_c).view(batch, tgt_len, self.dec_hidden))

        #transpose will make it non-contiguous
        attn_h = attn_h.transpose(0, 1).contiguous()
        align_vectors = align_vectors.transpose(0, 1).contiguous()
        #(tgt_len, batch, dim)
        return attn_h, align_vectors

    def score(self, h_t, h_s):
        """
        h_t (FloatTensor): batch x tgt_len x dim, inputs
        h_s (FloatTensor): batch x src_len x dim, context
        """
        tgt_batch, tgt_len, tgt_dim = h_t.size()
        src_batch, src_len, src_dim = h_s.size()

        h_t = h_t.view(tgt_batch*tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t)
        h_t = h_t.view(tgt_batch, tgt_len, tgt_dim)
        #(batch, d, s_len)
        h_s_ = h_s.transpose(1, 2)
        #(batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
        return torch.bmm(h_t, h_s_)
```

### DecoderRNN

```python
class DecoderRNN(nn.Module):
    """
    """
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, encoder_hidden=None, dropout_p=0.2, padding_idx=1, packed=True):
        super(DecoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.rnn = nn.GRU(input_size=embed_size, hidden_size=hidden_size, num_layers=n_layers)

        #h_t^T W h_s
        self.linear_out = nn.Linear(hidden_size, vocab_size)
        self.attn = GlobalAttention(encoder_hidden, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, inputs, hidden, context, context_lengths):
        """
        inputs: (tgt_len, batch_size, d)
        hidden: last hidden state from encoder
        context: (src_len, batch_size, hidden_size), outputs of encoder
        """
        #Teacher-forcing, not packed!
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)
        decoder_unpacked, decoder_hidden = self.rnn(embedded, hidden)
        #Calculate the attention.
        attn_outputs, attn_scores = self.attn(
            decoder_unpacked.transpose(0, 1).contiguous(),  # (len, batch, d) -> (batch, len, d)
            context.transpose(0, 1).contiguous(),         # (len, batch, d) -> (batch, len, d)
            context_lengths=context_lengths
        )
        #Don't need LogSoftmax with CrossEntropyLoss
        #the outputs are not normalized, and can be negative
        #Note that a mask is needed to compute the loss
        outputs = self.linear_out(attn_outputs)
        return outputs, decoder_hidden
```
