import torch
from tokenizers import Tokenizer, pre_tokenizers, models, trainers
from transformers import PreTrainedTokenizerFast
import datasets

ds = datasets.load_dataset('HuggingFaceFW/fineweb-edu',name='CC-MAIN-2025-26',streaming=True) # ~152B tokens
ds = ds.shuffle(buffer_size=100_000)
ds = ds['train'].take(50000)

tokenizer = Tokenizer(models.BPE(unk_token='<unk>'))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
# KeyError
# def iterator(dataset,batch_size=1000):
#    for i in range(0,50_000,batch_size):
#        yield dataset[i:i+batch_size]

def iterator(dataset,batch_size=1000):
    batch=[]
    for example in dataset:
        batch.append(example['text'])
        if len(batch)==batch_size:
            yield batch
            batch=[]
    if batch:
        yield batch

custom_iterator = iterator(ds,batch_size=1000)
trainer = trainers.BpeTrainer(vocab_size=49_267,special_tokens=['<unk>','<eos>','<bos>','<pad>'])
tokenizer.train_from_iterator(custom_iterator,trainer=trainer)
func_tokenizer = PreTrainedTokenizerFast(tokenizer,unk_token='<unk>',eos_token='<eos>',pad_token='<pad>',bos_token='<bos>')

func_tokenizer.save_pretrained('fineweb_tokenizer')
