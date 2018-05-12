
# naive-seq2seq


naive Sequence-to-Sequence Model with pytorch

Code From: [czs0x55aa/naive-seq2seq](https://github.com/czs0x55aa/naive-seq2seq)

### Encoder

```python
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, bidirectional=False):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        if bidirectional:
            self.bidirectional = 2
        else:
            self.bidirectional = 1

        self.nn_cell = nn.GRU(hidden_size, hidden_size, bidirectional=bidirectional)
        if USE_CUDA:
            self.nn_cell = self.nn_cell.cuda()

    def forward(self, input_embedded, hidden):
        output = input_embedded
        output, hidden = self.nn_cell(output, hidden)
        return output, hidden

    def initHidden(self):
        #(num_layers * num_directions, batch, hidden_size)
        init_hidden = Variable(torch.zeros(self.bidirectional, self.batch_size, self.hidden_size))
        if USE_CUDA:
            init_hidden = init_hidden.cuda()
        return init_hidden
```


### Decoder

```python
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.nn_cell = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        if USE_CUDA:
            self.nn_cell = self.nn_cell.cuda()
            self.out = self.out.cuda()

    def forward(self, input, hidden):
        output = input
        output, hidden = self.nn_cell(output, hidden)
        output = F.log_softmax(self.out(output.view(-1, self.hidden_size)))
        return output, hidden
```

### AttentionLayer

```python
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size, max_len):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, max_len)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, encoder_outputs, prev_hidden, decoder_input_embedded):
        attn_weights = F.softmax(
            self.attn(torch.cat((decoder_input_embedded, prev_hidden), 1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                encoder_outputs.transpose(0, 1))
        output = torch.cat((decoder_input_embedded.squeeze(0), attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output)
        return output
```

### Seq2Seq

```python
class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tar_vocab_size,
                    hidden_size,
                    max_input_steps=20, max_output_steps=20,
                    batch_size=30, dropout_p=0.2,
                    bidirectional=False):
        super(Seq2Seq, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.dropout_p = dropout_p
        self.max_input_steps = max_input_steps
        self.max_output_steps = max_output_steps
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.encode_out = nn.Linear(2 * self.hidden_size, self.hidden_size)

        self.attention = AttentionLayer(hidden_size, max_input_steps)
        self.init_embbeding(src_vocab_size, tar_vocab_size, hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.encoder = Encoder(src_vocab_size, hidden_size, batch_size, bidirectional=self.bidirectional)
        self.decoder = Decoder(hidden_size, tar_vocab_size)
        if USE_CUDA:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

    def init_embbeding(self, src_vocab_size, tar_vocab_size, hidden_size):
        self.src_embedding = nn.Embedding(src_vocab_size, hidden_size)
        self.tar_embedding = nn.Embedding(tar_vocab_size, hidden_size)
        #if USE_CUDA:
             #self.src_embedding.cuda()
             #self.tar_embedding.cuda()

    def __encode(self, encoder_input, inputs_length):
        #encoder_input_embedded size (batch_size, seq_len, dim)
        encoder_input_embedded = self.src_embedding(encoder_input)
        if USE_CUDA:
            encoder_input_embedded = encoder_input_embedded.cuda()
        #encoder_input_embedded size (seq_len, batch_size, dim)
        encoder_input_embedded = encoder_input_embedded.transpose(0, 1)
        #封装输入序列
        input_padded = pack_padded_sequence(encoder_input_embedded, inputs_length)
        #encoder_output size (seq_len, batch, hidden_size*directions)
        #encoder_hidden size (layers*directions, batch_size, hidden_size)
        encoder_output, encoder_hidden = self.encoder(input_padded, self.encoder.initHidden())
        encoder_output_packed = pad_packed_sequence(encoder_output)
        #if bidirectional is True
        if self.bidirectional:
            output = encoder_output_packed[0]
            output = self.encode_out(output.view(-1, 2 * self.hidden_size))
            output = output.view(-1, self.batch_size, self.hidden_size)
            encoder_hidden = torch.sum(encoder_hidden, 0)
        return output, encoder_hidden

    def __decode(self, encoder_outputs, encoder_hidden_output, decoder_input):
        decoder_outputs = Variable(torch.FloatTensor(self.max_output_steps, self.batch_size, self.tar_vocab_size))
        if USE_CUDA:
            decoder_outputs = decoder_outputs.cuda()
        #将编码器的输出隐层作为解码器的隐层初始值
        decoder_hidden = encoder_hidden_output
        if (decoder_input is not None) and (random.random() < 0.5):
            #teach mode
            #batch_input_embedded size (seq_len, batch_size, dim)
            batch_input_embedded = self.tar_embedding(decoder_input).transpose(0, 1)
            #dropout
            batch_input_embedded = self.dropout(batch_input_embedded)
            for time_step in range(self.max_output_steps):
                input_embedded = batch_input_embedded[time_step]
                #attention
                input_embedded = self.attention(encoder_outputs, decoder_hidden.squeeze(0), input_embedded)
                input_embedded = F.relu(input_embedded)
                decoder_output, decoder_hidden = self.decoder(input_embedded.unsqueeze(0), decoder_hidden)
                decoder_outputs[time_step] = decoder_output
        else:
            #without teach
            #解码器的第一个输入为EOS
            first_input = Variable(torch.LongTensor([EOS] * self.batch_size))
            if USE_CUDA:
                first_input = first_input.cuda()
            input_embedded = self.tar_embedding(first_input).unsqueeze(0)
            for time_step in range(self.max_output_steps):
                #dropout
                input_embedded = self.dropout(input_embedded)
                #attention
                input_embedded = self.attention(encoder_outputs, decoder_hidden.squeeze(0), input_embedded.squeeze(0))
                input_embedded = F.relu(input_embedded)
                decoder_output, decoder_hidden = self.decoder(input_embedded.unsqueeze(0), decoder_hidden)
                decoder_outputs[time_step] = decoder_output
                #将当前step的输出作为下一个step的输入
                input_embedded = self.tar_embedding(decoder_output.max(1)[1].transpose(0, 1))
        return decoder_outputs.transpose(0, 1)

    def forward(self, encoder_input, inputs_length, decoder_input=None):
        encoder_outputs, encoder_hidden = self.__encode(encoder_input, inputs_length)
        return self.__decode(encoder_outputs, encoder_hidden, decoder_input)

    def predict(self, encoder_input, inputs_length, decoder_input=None):
        predict_output = self.forward(encoder_input, inputs_length, decoder_input)
        _, res = torch.max(predict_output, 2)
        return res.squeeze(2)
```