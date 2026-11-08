import torch, math, datasets, gc
import torch.nn as nn
import numpy as np
from itertools import chain
from transformers import PreTrainedTokenizerFast
from model import Transformer
from torch.utils.data import DataLoader, IterableDataset
import bitsandbytes as bnb
from tqdm import tqdm
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark=True
EPOCHS = 5
BATCH_SIZE=64
ACCUMULATION_STEPS=8
DEBUG=False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SUPPORTED = (True if torch.cuda.get_device_capability()[0]>=7 else False)

tokenizer = PreTrainedTokenizerFast.from_pretrained('fineweb_tokenizer',pad_token='<pad>')
tokenizer.padding_side='right'
tokenizer.pad_token = '<pad>'

ds = datasets.load_dataset('HuggingFaceFW/fineweb-edu',name='CC-MAIN-2025-26',streaming=True,split='train') # ~152B tokens
ds = ds.shuffle(buffer_size=100_000)

to_keep = 1_750_000_000 # 48B is needed for a model of size ~240M parameters
current_tokens = 0

class ItDataset(IterableDataset):
    def __init__(self,gen_func,dataset_p):
        self.gen_func = gen_func
        self.dataset = dataset_p
    def __iter__(self):
        return self.gen_func(self.dataset)
SIZE=512 # 75% less VRAM footprint than 1024
# Stream and tokenize
# def gen_subset(dataset): # will add batching to ameliorate the CPU bottleneck during cloud training 
#     global current_tokens
#     buffer = []
#     batch_texts = []
#     batch_size=64
#     for sample in dataset:
#         tokens = tokenizer(sample['text'],add_special_tokens=True,return_attention_mask=False,max_length=1024)
#         num_tokens = len(tokens['input_ids'])
        
#         current_tokens+=num_tokens
#         buffer.extend(tokens['input_ids'])
        
#         while len(buffer)>=SIZE:
#             chunk = buffer[:SIZE]
#             buffer = buffer[SIZE:]

#             yield {
#                 'input_ids':chunk,
#                 'labels':chunk.copy()
#             }

#         if current_tokens>=to_keep:
#             break

SIZE = 512 

def gen_subset(dataset):
    global current_tokens
    buffer = []
    batch_texts = []
    batch_size = 64  # Group texts together for fast batch tokenization
    
    for sample in dataset:
        batch_texts.append(sample['text'])
        
        if len(batch_texts) == batch_size:
            # Tokenize the whole batch in one fast Rust call
            batch_outputs = tokenizer(
                batch_texts, 
                add_special_tokens=True, 
                return_attention_mask=False, 
                truncation=True, 
                max_length=1024
            )
            
            for input_ids in batch_outputs['input_ids']:
                num_tokens = len(input_ids)
                current_tokens += num_tokens
                buffer.extend(input_ids)
                
                while len(buffer) >= SIZE:
                    chunk = buffer[:SIZE]
                    buffer = buffer[SIZE:]
                    yield {
                        'input_ids': chunk,
                        'labels': chunk.copy()
                    }
            
            batch_texts = [] # Reset batch buffer

            if current_tokens >= to_keep:
                break
            
    # Process any leftover texts in the final partial batch
    if batch_texts and current_tokens < to_keep:
        batch_outputs = tokenizer(
            batch_texts, 
            add_special_tokens=True, 
            return_attention_mask=False, 
            truncation=True, 
            max_length=1024
        )
        for input_ids in batch_outputs['input_ids']:
            num_tokens = len(input_ids)
            current_tokens += num_tokens
            buffer.extend(input_ids)
            while len(buffer) >= SIZE:
                chunk = buffer[:SIZE]
                buffer = buffer[SIZE:]
                yield {
                    'input_ids': chunk,
                    'labels': chunk.copy()
                }
train_dataset = ItDataset(gen_subset,ds)

def custom_collate(batch): # for grouping
    input_ids = torch.tensor([item['input_ids'] for item in batch],dtype=torch.long)
    labels = torch.tensor([item['labels'] for item in batch],dtype=torch.long)
    return {'input_ids':input_ids,'labels':labels}

train_loader = DataLoader(train_dataset,batch_size=int(BATCH_SIZE/ACCUMULATION_STEPS),shuffle=False,num_workers=0,collate_fn=custom_collate,pin_memory=True)

model = Transformer()
model= model.to(device,non_blocking=True)
if hasattr(model,'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
    
# Check compute capability to see if torch.compile is supported
if torch.cuda.get_device_capability()[0]>=7:
    model = torch.compile(model)
if SUPPORTED:
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    optimizer = bnb.optim.AdamW8bit(model.parameters(),lr=3e-4,weight_decay=1e-5) # To stay under the VRAM envelope
else:
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5,fused=True) # For compatibility with devices other than CUDA enabled hardware
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=EPOCHS,eta_min=1e-6)

# Gradient accumulation to allow for larger effective batch sizes
best_loss = float('inf')
train_loss = 0.0
train_samples = 0.0
effective_batch_size = BATCH_SIZE/ACCUMULATION_STEPS

if SUPPORTED: # If volta or newer
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    dtype = torch.float32

for epoch in range(EPOCHS):
    model.train()
    epoch_loss=0.0
    num_batches=0
    loop = tqdm(train_loader,desc=f"Epoch: {epoch+1}/{EPOCHS}",leave=True)
    
    optimizer.zero_grad(set_to_none=True) # slight memory footprint reduction
    for step, batch in enumerate(loop):
        input_ids = batch['input_ids'].to(device,non_blocking=True)

        decoder_inp,decoder_tgt = input_ids[:,:-1], input_ids[:,1:]
        decoder_inp,decoder_tgt = decoder_inp.to(device,non_blocking=True), decoder_tgt.to(device,non_blocking=True)

        with torch.amp.autocast(device_type='cuda',dtype=dtype):
            preds = model(decoder_inp)
            loss = criterion(preds.reshape(-1,preds.shape[-1]), decoder_tgt.reshape(-1))
            unscaled_loss = loss.item()
            loss/=ACCUMULATION_STEPS
            
            if DEBUG:
                print("Preds shape:", preds.shape)          # Should be [batch_size, seq_len, 49267(vocab size)]
                print("Target max token ID:", decoder_tgt.max().item()) # Should be < 49267
                print("Logits min/max:", preds.min().item(), preds.max().item()) # sHould not be inf/nan or > 50
            
        if SUPPORTED:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        # Will only step the optimizer every N mini-batches
        if (step+1)%ACCUMULATION_STEPS==0:
            # Gradient clipping to prevent exploding gradients
            if SUPPORTED:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        epoch_loss+=unscaled_loss
        num_batches+=1
        loop.set_postfix(loss=f'{unscaled_loss:.4f}')

    # Handle any leftover gradients in case total batches aren't divisible by ACCUMULATION_STEPS
    if num_batches%ACCUMULATION_STEPS!=0:
        if SUPPORTED:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
 
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step(
                
            )
        optimizer.zero_grad(set_to_none=True)

    if (epoch+1)%5==0:
        torch.cuda.empty_cache()
        gc.collect()

    scheduler.step()
    avg_epoch_loss = epoch_loss/num_batches
    #print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_epoch_loss:.4f}")

    if avg_epoch_loss<best_loss:
        best_loss=avg_epoch_loss
        torch.save(model.state_dict(),f"model_best.pth")
        print(f'New best model saved with loss {best_loss:.4f}')
    
    torch.save(model.state_dict(),f'model_checkpoints/model_fineweb_epoch{epoch+1}.pth')
