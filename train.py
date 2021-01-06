
import argparse

def train_baseline_model():
    """ # Create dataset.
    dataset = COCODataset(mode='train', img_transform=train_img_transform())

    # Create train data loader
    pad_idx = dataset.vocab(PAD_TOKEN)
    def collate_fn(batch):
        batch = tuple(zip(*batch))
        imgs, captions, caption_lengths = batch[0][:], batch[1][:], batch[2][:]
        imgs = torch.stack(imgs)
        captions = pad_sequence(captions, batch_first=True, padding_value=pad_idx)
        caption_lengths = torch.stack(caption_lengths)
        return imgs, captions, caption_lengths
    # pin_memory: If True, the data loader will copy Tensors into CUDA pinned 
    # memory before returning them. If your data elements are a custom type,
    # or your collate_fn returns a batch that is a custom type.
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn)

    vocab = dataset.vocab """

def train_attention_model():
    pass

def train_bert_model():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--baseline_model', type=bool, default=False)
    parser.add_argument('--att_model', type=bool, default=False)
    parser.add_argument('--bert_model', type=bool, default=False)
    args = parser.parse_args()

    if args.baseline_model:
        print('Training baseline...')
        train_baseline_model()
    elif args.att_model:
        print('Training attention LSTM...')
        train_attention_model()
    elif args.bert_model: 
        print('Training BERT...')
        train_bert_model()
    else:
        print('No training...')

    
