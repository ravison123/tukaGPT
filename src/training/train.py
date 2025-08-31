import os
import torch
from src.data.get_batch import get_dataloader
from pathlib import Path
import numpy as np
from src.model.lm_torch import MarathiLM
import wandb
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import random

load_dotenv()


def train(model, train_dataloader, valid_dataloader, optimizer, scheduler, criterion, device, epochs,
          project_name='marathi_language_model', learning_rate=None, save_directory=None):
    wandb.login(key=os.getenv('WANDB_API_KEY'))

    model = model.to(device)
    wandb.init(
        project=project_name,
        config={
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size
        },
        name='marathi_lm_learning_rate={}'.format(learning_rate)
    )
    train_steps_per_epoch = len(train_dataloader)
    valid_steps_per_epoch = len(valid_dataloader)
    total_steps = train_steps_per_epoch * epochs
    wandb.watch(model)
    global_step = 0

    with tqdm(total=total_steps, desc="Training", unit="step") as pbar:
        for epoch in range(1, epochs + 1):
            model.train()
            total_train_loss = 0
            for i, (x, y) in enumerate(train_dataloader):
                if i != len(train_dataloader) - 1:
                    continue
                global_step = global_step + 1
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_train_loss += loss.item()
                wandb.log({"train/loss": loss, "lr": scheduler.get_last_lr()[0]}, step=global_step)
                pbar.set_postfix({"Training loss": f"{loss.item():.4f}"})
                pbar.update(1)

            avg_train_loss = total_train_loss / len(train_dataloader)

            try:
                checkpoint_path = os.path.join(save_directory, f"epoch_{epoch + 1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, checkpoint_path)
            except:
                print('Model logging failed')

            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                step_num = 0
                for x, y in valid_dataloader:
                    step_num = step_num + 1
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                    total_val_loss += loss.item()
                    valid_step = ((epoch - 1) * valid_steps_per_epoch) + step_num
                    wandb.log({"train/loss": loss}, step=valid_step)

            avg_val_loss = total_val_loss / len(valid_dataloader)

            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")


if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    vocab_size = 6000
    context_length = 512
    batch_size = 32
    d_model = 256
    num_heads = 4
    num_layers = 4
    d_ff = 1024
    EPOCHS = 10

    ROOT = Path(__file__).resolve().parent.parent.parent
    train_tokens_file = ROOT / "data" / "tukaram_gatha_train_encoded.npy"
    valid_tokens_file = ROOT / "data" / "tukaram_gatha_valid_encoded.npy"
    train_tokens = np.load(train_tokens_file)
    valid_tokens = np.load(valid_tokens_file)

    train_dataloader = get_dataloader(train_tokens, context_length, batch_size, shuffle = True)
    valid_dataloader = get_dataloader(valid_tokens, context_length, batch_size, shuffle=True)

    # for i, batch in enumerate(train_dataloader):
    #     if i == len(train_dataloader) - 1:
    #         print('The last batch')
    #     if batch[0].numel() == 0:
    #         print(f"Empty x batch at {i}")
    #     if batch[1].numel() == 0:
    #         print(f"Empty x batch at {i}")

    model = MarathiLM(vocab_size, context_length, d_model=d_model, num_heads=num_heads, n_layers=num_layers, d_ff=d_ff, device='mps')
    total_params = 0
    for param in model.parameters():
        total_params = total_params + param.numel()

    print('Parameters of the model are: {}'.format(total_params / 1e+06))

    save_directory = ROOT / "checkpoints" / "7M_Model"

    learning_rates = [3e-03]
    total_steps = len(train_dataloader) * EPOCHS
    for learning_rate in learning_rates:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(0.1 * total_steps),
                                                    num_training_steps=total_steps)
        criterion = torch.nn.CrossEntropyLoss()

        project_name = 'marathi_llm_{}M_params_LR_tuning'.format(total_params/1e+06)
        train(model, train_dataloader, valid_dataloader, optimizer, scheduler, criterion, epochs=EPOCHS, device="mps",
              learning_rate=learning_rate, project_name=project_name, save_directory=save_directory)