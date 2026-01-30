import numpy as np
import torch
import time
import json
from sklearn.metrics import r2_score

import torch.distributed as dist

import warnings
warnings.filterwarnings("ignore")

def ssl_train_validate(is_distributed, model, optimizer, loss_function, epochs, patience, rank, experiment_dir, train_dataloader, target_train_dataloader, val_dataloader=None):
    
    start_total = time.time()

    mean_r2_max = -100
    epoch_best_model = 0
    
    train_metrics_names = ["train_loss", "train_r2"]
    val_metrics_names = ["val_loss", "val_r2"]

    train_metrics = {name: [] for name in train_metrics_names}
    val_metrics = {name: [] for name in val_metrics_names}
    
    for epoch in range(epochs):

        mean_r2_epoch_cumulative = 0
    
        if rank == 0 or is_distributed == False:
            print('##############################################', flush=True)
            print('Starting epoch ' + str(epoch + 1) + '/' + str(epochs) + '...', flush=True)
            print('Training...', flush=True)
            start_epoch = time.time()
        
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch) 

        model.train()

        total_loss_epoch = 0
        total_r2_epoch = 0

        # Iterando nos batches de treino
        for idx_batch, (input_batch, target_batch) in enumerate(zip(train_dataloader, target_train_dataloader)):
            
            optimizer.zero_grad()
            output_batch = model(input_batch)
            target_batch = target_batch.float()
            loss = loss_function(output_batch, target_batch)
            loss.backward()
            optimizer.step()

            '''
            if idx_batch == 0:
                np.save(f'{experiment_dir}/{epoch}_input', input_batch[0].detach().cpu().numpy())
                np.save(f'{experiment_dir}/{epoch}_output', output_batch[0].detach().cpu().numpy())
            '''
            
            loss_batch = loss.detach().item()
            total_loss_epoch += loss_batch

            target_batch = target_batch.detach().cpu().numpy().ravel()
            output_batch = output_batch.detach().cpu().numpy().ravel()
            
            r2_batch = r2_score(target_batch, output_batch) 
            total_r2_epoch += r2_batch

        mean_loss_epoch = total_loss_epoch / (idx_batch + 1)
        mean_r2_epoch = total_r2_epoch / (idx_batch + 1)

        mean_r2_epoch_cumulative += mean_r2_epoch

        if rank == 0 or is_distributed == False:
            print(f"\tLoss train: {mean_loss_epoch:.5f}", flush=True)
            print(f"\tR2 train: {mean_r2_epoch:.5f}", flush=True)
        
        metrics_values = [mean_loss_epoch, mean_r2_epoch]

        for i in range(len(train_metrics_names)):
            name_metric = train_metrics_names[i]
            value = metrics_values[i]
            train_metrics[name_metric].append(value)

        if (rank == 0 or is_distributed == False) and ((epoch + 1) % 5) == 0:
            with open(f"{experiment_dir}/SSL_all_metrics_train.json", "w") as file:
                json.dump(train_metrics, file, indent=4)

        ##################################################################

        if (rank == 0 or is_distributed == False) and val_dataloader is not None:
            print('Validation...')
            
            if is_distributed:
                train_dataloader.sampler.set_epoch(epoch) 
            
            model.eval()

            total_loss_epoch = 0
            total_r2_epoch = 0

            with torch.no_grad():
                
            # Iterando nos batches de validação
                for idx_batch, input_batch in enumerate(val_dataloader):
        
                    optimizer.zero_grad()
                    output_batch = model(input_batch)
  
                    loss = loss_function(output_batch, input_batch)                    
                    loss_batch = loss.detach().item()
                    total_loss_epoch += loss_batch
        
                    input_batch= input_batch.detach().cpu().numpy().ravel()
                    output_batch = output_batch.detach().cpu().numpy().ravel()
                    
                    r2_batch = r2_score(input_batch, output_batch) 
                    total_r2_epoch += r2_batch
        
                mean_loss_epoch = total_loss_epoch / (idx_batch + 1)
                mean_r2_epoch = total_r2_epoch / (idx_batch + 1)
                mean_r2_epoch_cumulative += mean_r2_epoch
        
                if rank == 0 or is_distributed == False:
                    print(f"\tLoss validation: {mean_loss_epoch:.5f}", flush=True)
                    print(f"\tR2 validation: {mean_r2_epoch:.5f}", flush=True)
                
                metrics_values = [mean_loss_epoch, mean_r2_epoch]
        
                for i in range(len(val_metrics_names)):
                    name_metric = val_metrics_names[i]
                    value = metrics_values[i]
                    val_metrics[name_metric].append(value)

                if (rank == 0 or is_distributed == False) and ((epoch + 1) % 5) == 0:
                    with open(f"{experiment_dir}/SSL_all_metrics_validation.json", "w") as file:
                        json.dump(val_metrics, file, indent=4)
                        
                ############# salvamento do melhor modelo (menor loss de validação) ###############

                mean_r2_epoch_cumulative = mean_r2_epoch_cumulative / 2
                
                if (rank == 0 or is_distributed == False) and (mean_r2_epoch_cumulative >= mean_r2_max):
                    epoch_best_model = epoch
                    mean_r2_max = mean_r2_epoch_cumulative
                    # Salvando modelo e otimizador
                    torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict()},
                               f"{experiment_dir}/SSL_complete_model.pth")

        if rank == 0 or is_distributed == False:
            print(f'Time: {time.time() - start_epoch:,.2f} sec', flush=True)

        if epoch == epoch_best_model + patience:
            print('----------------------------------------------', flush=True)
            print('TRAINING INTERRUPTED EARLIER, MODEL STOPPED IMPROVING VALIDATION.')
            break
     
    if rank == 0 or is_distributed == False:
        print('----------------------------------------------', flush=True)
        print(f'Maximum mean R2: {mean_r2_max:,.5f}. Epoch: {epoch_best_model + 1}', flush=True)
        print('----------------------------------------------', flush=True)
        print(f'Total time: {(time.time() - start_total)/60:,.1f} min')
        print('----------------------------------------------', flush=True)
