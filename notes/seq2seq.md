
## Sequence to Sequence

> March 19

### 1. [Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://arxiv.org/pdf/1703.07015.pdf)

Source code: [https://github.com/laiguokun/LSTNet](https://github.com/laiguokun/LSTNet)

Long- and Short-term Time-series network (LSTNet)

- LSTNet uses the Convolution Neural Network (CNN) to extract short-term local dependency patterns among variables, 

- and the Recurrent Neural Network (RNN) to discover long-term patterns and trends.

- A traditional autoregressive linear model to provide more sensitive, when/if the non-linear model is not sufficiently sensitive to the scale changes in input data.


### 2. [Hierarchical Multiscale Recurrent Neural Networks](https://arxiv.org/pdf/1609.01704.pdf)

Source code: [https://github.com/n-s-f/hierarchical-rnn](https://github.com/n-s-f/hierarchical-rnn)

Key words: hierarchical and temporal representation, multiscale,

- capture the latent hierarchical structure in the sequence by encoding the temporal dependencies with different timescales using a novel update mehanism.

- discover underlying hierarchical structure in the sequences without using explicit boundary.

- evaluate the model on character-level language modelling and handwriting sequence generation.

- The multiscale RNNs group hidden units into multiple modules of different timescales.

- It is important for an RNN to dynamically adapt its timescales to the particulars of the input entities of various length.

- three operations: UPDATE, COPY and FLUSH 


### 3. [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/pdf/1803.04831.pdf)

Source code: [https://github.com/batzner/indrnn](https://github.com/batzner/indrnn)

- Can be easily regulated to prevent the gradient exploding and vanishing problems while allowing the network to learn long-term dependencies.

- Can work with non-saturated activation functions such as relu (rectfied linear unit) and be still trained robustly.

- Multiple IndRNNs can be stacked to contruct a network that is deeper than the existing RNNs.


> Mar 21

### 4. [Deep Speech 2: End-to-End Speech Recognition in English and Mandarin](https://arxiv.org/pdf/1512.02595v1.pdf)

Source code: [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)

```
DeepSpeech(
  (conv): Sequential(
    (0): Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
    (2): Hardtanh(min_val=0, max_val=20, inplace)
    (3): Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1))
    (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True)
    (5): Hardtanh(min_val=0, max_val=20, inplace)
  )
  (rnns): Sequential(
    (0): BatchRNN(
      (rnn): GRU(672, 800, bias=False, bidirectional=True)
    )
    (1): BatchRNN(
      (batch_norm): SequenceWise (
      BatchNorm1d(800, eps=1e-05, momentum=0.1, affine=True))
      (rnn): GRU(800, 800, bias=False, bidirectional=True)
    )
    (2): BatchRNN(
      (batch_norm): SequenceWise (
      BatchNorm1d(800, eps=1e-05, momentum=0.1, affine=True))
      (rnn): GRU(800, 800, bias=False, bidirectional=True)
    )
    (3): BatchRNN(
      (batch_norm): SequenceWise (
      BatchNorm1d(800, eps=1e-05, momentum=0.1, affine=True))
      (rnn): GRU(800, 800, bias=False, bidirectional=True)
    )
    (4): BatchRNN(
      (batch_norm): SequenceWise (
      BatchNorm1d(800, eps=1e-05, momentum=0.1, affine=True))
      (rnn): GRU(800, 800, bias=False, bidirectional=True)
    )
  )
  (fc): Sequential(
    (0): SequenceWise (
    Sequential(
      (0): BatchNorm1d(800, eps=1e-05, momentum=0.1, affine=True)
      (1): Linear(in_features=800, out_features=29, bias=False)
    ))
  )
  (inference_softmax): InferenceBatchSoftmax(
  )
)
Number of parameters: 38067968
```

> Mar 27

### 5. [Twin Networks: Matching the Future for Sequence Generation](https://arxiv.org/abs/1708.06742)

Source code: [https://github.com/dmitriy-serdyuk/twin-net/](https://github.com/dmitriy-serdyuk/twin-net/)

- Train a "backward" recurrent network to generate a given sequence in reverse order

- Encourage states of the forward model to predict cotemporal states of backward model

- The backward network is used only during training, and plays no role during sampling or inference

- Eases modeling of long-term dependencies by implicitly forcing the forward states to hold information about the longer-term future (as contained in the backward states)


> Mar 28

### 6. [Cold Fusion: Training Seq2Seq Models Together with Language Models](https://arxiv.org/pdf/1708.06426.pdf)

- Sequence-to-sequence (Seq2Seq) models with attention

- Cold Fusion method, leverages a pre-trained language model during training, and show its effectiveness on the speech recognition task.

- better utilize language information enjoying faster convergence and better generalization

- almost complete transfer to a new domain while using less than 10% of labeled training data.

- Seq2Seq models learn to generate a variable-length sequence of tokens (e.g. texts) from a variable-lenth sequence of input data(e.g. speech of the same texts in another language).

- With a sufficiently large labeled dataset, vanilla Seq2Seq can model sequential mapping well, but it is often augmented with a language model to further improve the fluency of the generated text.

### 7. [Segmental Recurrent Neural Networks](https://arxiv.org/pdf/1511.06018.pdf)

Source code: [https://github.com/zhan-xiong/segrnn](https://github.com/zhan-xiong/segrnn)

- a joint probability distribution over segmentations of the input and labelings of the segments.

- Representations of the input segments are computed by encoding their constituent tokens unsing bidirectional recurrent neural nets,

- these "segment embeddings" are used to define compatibility scores with output labels.

- These local compatibility scores are integrated using a global semi-Markov conditional random field.

- Fully supervised training, segment boundaries and labels are observed, partially supervised training, segment boundaries are latent.

- Breaking an input sequence into contiguous, arbitrary-length segments while labeling each segment.

- two tools: representation learning and structured prediction


> Mar 31


### 8. [Segmental Recurrent Neural Networks for End-to-end Speech Recognition](https://arxiv.org/pdf/1603.00223.pdf)

- connect the segmental conditional random filed (CRF) with a RNN used for feature extraction

- it does not rely on an external system to provide features or segmentation boundaries

- this model marginalises out all the possible segmentations

- features are extracted from the RNN trained together with the segmental CRF

- this model is self-contained and can be trained end-to-end.

- Speech recognition is a typical sequence to sequence transduction problem, given a sequence of acoustic observations, decodes the corresponding sequence of words or phonemes.

- A key component is the acoustic model, which computes the conditional probability of the output sequence given the input sequence.

- challenging, variable lengths of the input and output sequences

- The hidden Markov model (HMM) converts this sequence-level classification task into a frame-level classification problem, it relies on the conditional independence assumption and first-order Markov rule - the well-known weaknesses of HMMs.

- The connectionist temporal classification (CTC) defines the lost function directly to maximise the conditional probability of the output sequence given the input sequence.

- CTC simplifies the sequence-level error function by a product of the frame-level error functions (i.e., independence assumption), which means it essentially still does frame-level classification.

- It also requried the lengths of the input and output sequence to be the same, which is inappropriate for speech recognition.

- CTC deals with this problem by replicating the output labels so that a consecutive frames may correspond to the same output lable or a blank token.

- A key difference of Attention-based RNNs from HMMs and CTCs is that this approach does not apply the conditional independence assumption to the input sequence.

- It maps the variable-length input sequence into a fixed-size vector representation at each decoding step by an attention-based scheme. It then generates the output sequence using an RNN conditioned on the vector representation from the source sequence.

- This attentive scheme suits the machine translation task well, because there may be no clear alignment between the source and target sequence for many language pairs.


### 9. [Listen, Attend and Spell](https://arxiv.org/pdf/1508.01211v2.pdf)

Source code: [https://github.com/XenderLiu/Listen-Attend-and-Spell-Pytorch](https://github.com/XenderLiu/Listen-Attend-and-Spell-Pytorch)

- Unlike DNN-HMM models, this model learns all the components of a speech recognizer jointly

- Two components: a listener and speller. The listener is a pyramidal recurrent network encoder that accepts filter bank spectra as inputs. The speller is an attention-based recurrent network decoder that emits characters as outputs.

- The network produces character sequences without making any independence assumptions between the characters.

- CTC assumes that the label outputs are conditionally independent of each other; whereas the sequence to sequence approach has only been applied to phoneme sequences, and not trained end-to-end for speech recognition.

- This Listen, Attend and Spell (LAS) model learns to transcribe an audio sequence signal to a word sequence, one character at a time.

- LAS is based on the sequence to sequence learning framework with attention

- The listener is a pyramidal RNN that converts low level speech signals into higher level features. The speller is an RNN that converts these higher level features into output utterances by specifying a probability distribution over sequences of characters using the attention mechanism. The listener and speller are trained jointly.

- The pyramidal RNN model for the listener, which reduces the number of times steps that the attention model has to extract relevant information from.

- Another advantage is that it is able to generate multiple spelling variants naturally.

- Without the attention mechanism, the model overfits the training data significantly, in spite of our large training set of three million utterances - it memorizes the training transcripts without paying attention to the acoustics.

- Without the pyramid structure in the encoder side, our model coverges too slowly - even after a month of training, the error rates were significantly higher.

- Both of these problems arise because the acoustic signals can have hundreds to thousands of frames which makes it difficult to train the RNNs.

- Classification problems: mapping a fixed-length vector to an output category, for structured problem, such as mapping one variable-length sequence to another variable-length sequence, neural networks have to be combined with other sequential models such as Hidden Markov Models (HMMs) and Conditional Random Fields (CRFs).

- A drawback of these combining approaches is that the resulting models cannot be easily trained end-to-end and they make simplistic assumptions about the probability distribution of the data.

- Seq2Seq learning is a framework that attempts to address the problem of learning variable-length input and output sequences. It uses an encoder RNNs to map the sequential variable-length input into a fixed-length vector. A decoder RNN then uses this vector to produce the variable-length output sequence, one token at a time. 

- During training, the model feeds the groundtruth labels as inputs to the decoder. During inference, the model performs a beam search to generate suitable candidates for next step predications.

- Seq2Seq models can be improved significantly by the use of an attention mechanism that provides the decoder RNN more information when it produces the output tokens.

- At each output step, the last hidden state of the decoder RNN is used to generate an attention vector over the input sequence of the encoder.

- The attention vector is used to propagate information from the encoder to the decoder at every time step, instead of just once, as with the original sequence to sequence model.

- This attention vector can be thought of as skip connections that allow the information and the gradients to flow more effectiavely in an RNN.

