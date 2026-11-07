import torch, math, datasets
import numpy as np
from itertools import chain
from transformers import PreTrainedTokenizerFast
from model import Transformer
from torch.utils.data import DataLoader, IterableDataset
import bitsandbytes as bnb

EPOCHS = 50
device = torch.device('cuda' if troch.cuda.is_available() else 'cpu')

tokenizer = PreTrainedTokenizerFast.from_pretrained('fineweb_tokenizer',pad_token='<pad>')
tokenizer.padding_side='right'
tokenizer.pad_token = '<eos>'

ds = datasets.load_dataset('HuggingFaceFW/fineweb-edu',subset='CC-MAIN-2025-26',streaming=True) # ~152B tokens
ds = ds.shuffle(buffer_size=100_000)

to_keep = 24_000_0000_000 # 24B is actually needed for a model of size 120M parameters
current_tokens = 0

class ItDataset(nn.Module):
    def __init__(self,gen_func):
        self.gen_func = gen_func
    def forward(self):
        self.gen_func()

# Stream and tokenize
def gen_subset(dataset):
    global current_tokens
    buffer = []
    for sample in dataset:
        tokens = tokenizer(sample['text'],add_special_tokens=True,return_attention_mask=False,max_length=1024)
        num_tokens = len(tokens['input_ids'])
        
        current_tokens+=num_tokens
        buffer.extend(tokens['input_ids'])
        
        while line(buffer)>=SIZE:
            chunk = buffer[:SIZE]
            buffer = buffer[SIZE:]

            yield {
                'input_ids':chunk,
                'labels':chunk.copy()
            }

        if current_tokens>=to_keep:
            break

train_dataset = ItDataset(gen_subset)

def custom_collate(batch): # for grouping
    input_ids = torch.tensor([item['input_ids'] for item in batch],dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch],dtype=torch.long)
    return {'input_ids':input_ids,'labels':labels}

train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True,collate_fn=custom_collate,num_workers=0,pin_memory=True)

model = Transformer()
# Check compute capability to see if torch.compile is supported
if torch.cuda.get_device_capability()[0]>=7:
    model = torch.compile(model)
else:
    continue

model= model.to(device,non_blocking=True)

scaler = torch.amp.GradScaler(device)
optimizer = bnb.optim.AdamW8bit(model.parameters,lr=1e-4,weight_decay=1e-5) # To stay under the VRAM envelope
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)

train_loss = 0.0
train_samples = 0.0
for epoch in range(EPOCHS):
    model.train()
    epoch_loss
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device,non_blocking=True)

        decoder_inp,decoder_tgt = input_ids[:,:-1], input_ids[:,1:]
        decoder_inp,decoder_tgt = decoder_inp.to(device,non_blocking=True), decoder_tgt.to(device,non_blocking=True)

        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        with torch.amp.autocast(device_type='cuda',dtype=dtype):
            preds = model(decoder_inp)
            loss = criterion(preds.reshape(-1,preds.shape[-1]), decoder_tgt.reshape(-1))
            
        optimizer.zero_grad(set_to_none=True) # slight memory footprint reduction
        scaler.scale(loss).backward()

        # Gradient clipping to prevent exploding gradients
        scaler.unscale_(optimizer)
        torch.utils.clip_grad_norm(model.parameters(),max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        batch_size = input_ids.size(0)
        epoch_loss+=loss.item()*batch_size
        train_samples+=batch_size

    scheduler.step()
    avg_epoch_loss = epoch_loss/len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_epoch_loss:.4f}")

    torch.save(model.state_dict(),f'model_fineweb_epoch{epoch+1}')