import argparse
import random
import os

import numpy as np

import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.utils.data

import src.utils as utils
import src.dataset as dataset

import crnn.seq2seq as crnn

cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--train_list', type=str, help='path to train dataset list file')
parser.add_argument('--eval_list', type=str, help='path to evalation dataset list file')
parser.add_argument('--num_workers', type=int, default=4, help='number of data loading num_workers')
parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
parser.add_argument('--img_height', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--img_width', type=int, default=280, help='the width of the input image to network')
parser.add_argument('--hidden_size', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for Critic, default=0.00005')
parser.add_argument('--encoder', type=str, default='', help="path to encoder (to continue training)")
parser.add_argument('--decoder', type=str, default='', help='path to decoder (to continue training)')
parser.add_argument('--model', default='./model/', help='Where to store samples and models')
parser.add_argument('--random_sample', default=True, action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--teaching_forcing_prob', type=float, default=0.5, help='where to use teach forcing')
parser.add_argument('--max_width', type=int, default=71, help='the width of the feature map out from cnn')
cfg = parser.parse_args()
print(cfg)

# load alphabet
with open('./data/char_std_5990.txt') as f:
    data = f.readlines()
    alphabet = [x.rstrip() for x in data]
    alphabet = ''.join(alphabet)

# define convert bwteen string and label index
converter = utils.ConvertBetweenStringAndLabel(alphabet)

# len(alphabet) + SOS_TOKEN + EOS_TOKEN
num_classes = len(alphabet) + 2

def train(image, text, encoder, decoder, criterion, train_loader, teach_forcing_prob=1):
    # optimizer
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=cfg.learning_rate, betas=(0.5, 0.999))

    # loss averager
    loss_avg = utils.Averager()

    for epoch in range(cfg.num_epochs):
        train_iter = iter(train_loader)

        for i in range(len(train_loader)):
            cpu_images, cpu_texts = train_iter.next()
            batch_size = cpu_images.size(0)

            for encoder_param, decoder_param in zip(encoder.parameters(), decoder.parameters()):
                encoder_param.requires_grad = True
                decoder_param.requires_grad = True
            encoder.train()
            decoder.train()

            target_variable = converter.encode(cpu_texts)
            utils.load_data(image, cpu_images)

            # CNN + BiLSTM
            encoder_outputs = encoder(image)
            target_variable = target_variable.cuda()
            # start decoder for SOS_TOKEN
            decoder_input = target_variable[utils.SOS_TOKEN].cuda()
            decoder_hidden = decoder.initHidden(batch_size).cuda()
            
            loss = 0.0
            teach_forcing = True if random.random() > teach_forcing_prob else False
            if teach_forcing:
                for di in range(1, target_variable.shape[0]):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target_variable[di])
                    decoder_input = target_variable[di]
            else:
                for di in range(1, target_variable.shape[0]):
                    decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output, target_variable[di])
                    topv, topi = decoder_output.data.topk(1)
                    ni = topi.squeeze()
                    decoder_input = ni
            encoder.zero_grad()
            decoder.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            loss_avg.add(loss)

            if i % 10 == 0:
                print('[Epoch {0}/{1}] [Batch {2}/{3}] Loss: {4}'.format(epoch, cfg.num_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

        # save checkpoint
        torch.save(encoder.state_dict(), '{0}/encoder_{1}.pth'.format(cfg.model, epoch))
        torch.save(decoder.state_dict(), '{0}/decoder_{1}.pth'.format(cfg.model, epoch))


def evaluate(image, text, encoder, decoder, data_loader, max_eval_iter=100):

    for e, d in zip(encoder.parameters(), decoder.parameters()):
        e.requires_grad = False
        d.requires_grad = False

    encoder.eval()
    decoder.eval()
    val_iter = iter(data_loader)

    n_correct = 0
    n_total = 0
    loss_avg = utils.Averager()

    for i in range(min(len(data_loader), max_eval_iter)):
        cpu_images, cpu_texts = val_iter.next()
        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)

        target_variable = converter.encode(cpu_texts)
        n_total += len(cpu_texts[0]) + 1

        decoded_words = []
        decoded_label = []
        encoder_outputs = encoder(image)
        target_variable = target_variable.cuda()
        decoder_input = target_variable[0].cuda()
        decoder_hidden = decoder.initHidden(batch_size).cuda()

        for di in range(1, target_variable.shape[0]):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi.squeeze(1)
            decoder_input = ni
            if ni == utils.EOS_TOKEN:
                decoded_label.append(utils.EOS_TOKEN)
                break
            else:
                decoded_words.append(converter.decode(ni))
                decoded_label.append(ni)

        for pred, target in zip(decoded_label, target_variable[1:,:]):
            if pred == target:
                n_correct += 1

        if i % 10 == 0:
            texts = cpu_texts[0]
            print('pred: {}, gt: {}'.format(''.join(decoded_words), texts))

    accuracy = n_correct / float(n_total)
    print('Test loss: {}, accuray: {}'.format(loss_avg.val(), accuracy))


def main():
    if not os.path.exists(cfg.model):
        os.makedirs(cfg.model)

    # create train dataset
    train_dataset = dataset.TextLineDataset(text_line_file=cfg.train_list, transform=None)
    sampler = dataset.RandomSequentialSampler(train_dataset, cfg.batch_size)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False, sampler=sampler, num_workers=int(cfg.num_workers),
        collate_fn=dataset.AlignCollate(img_height=cfg.img_height, img_width=cfg.img_width))

    # create test dataset
    test_dataset = dataset.TextLineDataset(text_line_file=cfg.eval_list, transform=dataset.ResizeNormalize(img_width=cfg.img_width, img_height=cfg.img_height))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=int(cfg.num_workers))

    # create crnn/seq2seq/attention network
    encoder = crnn.Encoder(channel_size=3, hidden_size=cfg.hidden_size)
    # for prediction of an indefinite long sequence
    decoder = crnn.Decoder(hidden_size=cfg.hidden_size, output_size=num_classes, dropout_p=0.1, max_lrngth=cfg.max_width)
    print(encoder)
    print(decoder)
    encoder.apply(utils.weights_init)
    decoder.apply(utils.weights_init)
    if cfg.encoder:
        print('loading pretrained encoder model from %s' % cfg.encoder)
        encoder.load_state_dict(torch.load(cfg.encoder))
    if cfg.decoder:
        print('loading pretrained encoder model from %s' % cfg.decoder)
        decoder.load_state_dict(torch.load(cfg.decoder))

    # create input tensor
    image = torch.FloatTensor(cfg.batch_size, 3, cfg.img_height, cfg.img_width)
    text = torch.LongTensor(cfg.batch_size)

    criterion = torch.nn.NLLLoss()

    assert torch.cuda.is_available(), "Please run \'train.py\' script on nvidia cuda devices."
    encoder.cuda()
    decoder.cuda()
    image = image.cuda()
    text = text.cuda()
    criterion = criterion.cuda()

    # train crnn
    train(image, text, encoder, decoder, criterion, train_loader, teach_forcing_prob=cfg.teaching_forcing_prob)

    # do evaluation after training
    evaluate(image, text, encoder, decoder, test_loader, max_eval_iter=100)


if __name__ == "__main__":
    main()
