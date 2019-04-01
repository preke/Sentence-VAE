import os
import json
import torch
import argparse

from model import SentenceVAE
from utils import to_var, idx2word, interpolate


def decoder(model, z):
    hidden = model.latent2hidden(z)

    if model.bidirectional or model.num_layers > 1:
        # unflatten hidden state
        hidden = hidden.view(model.hidden_factor, batch_size, model.hidden_size)
    else:
        hidden = hidden.unsqueeze(0)

    # decoder input
    if model.word_dropout_rate > 0:
        # randomly replace decoder input with <unk>
        prob = torch.rand(input_sequence.size())
        if torch.cuda.is_available():
            prob=prob.cuda()
        prob[(input_sequence.data - model.sos_idx) * (input_sequence.data - model.pad_idx) == 0] = 1
        decoder_input_sequence = input_sequence.clone()
        decoder_input_sequence[prob < model.word_dropout_rate] = model.unk_idx
        input_embedding = model.embedding(decoder_input_sequence)
    input_embedding = model.embedding_dropout(input_embedding)
    packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

    # decoder forward pass
    outputs, _ = model.decoder_rnn(packed_input, hidden)

    # process outputs
    padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
    padded_outputs = padded_outputs.contiguous()
    _,reversed_idx = torch.sort(sorted_idx)
    padded_outputs = padded_outputs[reversed_idx]
    b,s,_ = padded_outputs.size()

    # project outputs to vocab
    logp = nn.functional.log_softmax(model.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
    logp = logp.view(b, s, model.embedding.num_embeddings)



def main(args):

    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s"%(args.load_checkpoint))

    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    valid_data = json.load('/home/wenzhy/Sentence-VAE/dumps/2019-Apr-01-08:37:49/valid_E9.json')
    sentences = valid_data['target_sents'][:10]
    zs = valid_data['z'][:10]
    print(sentences)
    print(zs)
    
    # samples, z = model.inference(n=args.num_samples)
    # print('----------SAMPLES----------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')
    # print(z.shape)
    # decoder(model, z)


    

    # z1 = torch.randn([args.latent_size]).numpy()
    # z2 = torch.randn([args.latent_size]).numpy()
    # z = to_var(torch.from_numpy(interpolate(start=z1, end=z2, steps=8)).float())
    # samples, _ = model.inference(z=z)
    # print(samples.size())
    # print('-------INTERPOLATION-------')
    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
