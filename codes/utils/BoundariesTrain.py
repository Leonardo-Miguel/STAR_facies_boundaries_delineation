from utils.Metrics import binary_metrics
import numpy as np
import torch
import time
import json
from sklearn.metrics import r2_score

import torch.distributed as dist

import warnings
warnings.filterwarnings("ignore")

def train_validate(is_distributed, model, optimizer, loss_function, epochs, patience, rank, experiment_dir, train_dataloader, val_dataloader=None):

    start_total = time.time()

    min_bdte_validation = 10e6
    epoch_best_model = 0
    
    train_metrics_names = ["train_loss", "train_bdte", "train_acc", "train_precision", "train_recall", "train_fallout", "train_iou", "train_f1",
                           "weighted_train_acc", "weighted_train_precision", "weighted_train_recall", "weighted_train_fallout", "weighted_train_iou", "weighted_train_f1"]
    
    val_metrics_names = ["val_loss", "val_bdte", "val_acc", "val_precision","val_recall", "val_fallout", "val_iou", "val_f1",
                         "weighted_val_acc", "weighted_val_precision","weighted_val_recall", "weighted_val_fallout", "weighted_val_iou", "weighted_val_f1"]

    train_metrics = {name: [] for name in train_metrics_names}
    val_metrics = {name: [] for name in val_metrics_names}

    for epoch in range(epochs):
    
        if rank == 0 or is_distributed == False:
            print('##############################################', flush=True)
            print('Starting epoch ' + str(epoch + 1) + '/' + str(epochs) + '...', flush=True)
            print('Training...', flush=True)
            start_epoch = time.time()
        
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch) 

        model.train()

        total_loss_epoch = 0
        total_bdte_epoch = 0
        total_acc_epoch = 0
        total_precision_epoch = 0
        total_recall_epoch = 0
        total_fallout_epoch = 0
        total_iou_epoch = 0
        total_f1_epoch = 0
        total_weighted_acc_epoch = 0
        total_weighted_precision_epoch = 0
        total_weighted_recall_epoch = 0
        total_weighted_fallout_epoch = 0
        total_weighted_iou_epoch = 0
        total_weighted_f1_epoch = 0
    
        # Iterando nos batches de treino
        for idx_batch, (input_batch, ground_truth_batch) in enumerate(train_dataloader):

            optimizer.zero_grad()
            output_batch = model(input_batch)
            ground_truth_batch = ground_truth_batch.float()
            loss = loss_function(output_batch, ground_truth_batch)
            loss.backward()
            optimizer.step()
            
            loss_batch = loss.detach().item()
            total_loss_epoch += loss_batch

            # IMPORTANTE! Como as saídas são em logits, é necessário binarizar com a sigmoide para calcular corretamente as métricas
            output_batch = torch.sigmoid(output_batch)
            output_batch = (output_batch > 0.5).to(torch.uint8)

            '''
            if idx_batch == 0:
                np.save(f'{experiment_dir}/{epoch}_input', input_batch[0].detach().cpu().numpy())
                np.save(f'{experiment_dir}/{epoch}_output', output_batch[0].detach().cpu().numpy())
            '''

            # Métricas convencionais, sem expansão das bordas
            bdte_batch, acc_batch, precision_batch, recall_batch, fallout_batch, iou_batch, f1_batch = binary_metrics(ground_truth_batch, output_batch, weighted_metrics=False)
            total_bdte_epoch += bdte_batch
            total_acc_epoch += acc_batch
            total_precision_epoch += precision_batch
            total_recall_epoch += recall_batch
            total_fallout_epoch += fallout_batch
            total_iou_epoch += iou_batch
            total_f1_epoch += f1_batch

            # Métricas ponderadas
            bdte_batch, weighted_acc_batch, weighted_precision_batch, weighted_recall_batch, weighted_fallout_batch, weighted_iou_batch, weighted_f1_batch = binary_metrics(ground_truth_batch, output_batch, weighted_metrics=True)
            total_weighted_acc_epoch += weighted_acc_batch
            total_weighted_precision_epoch += weighted_precision_batch
            total_weighted_recall_epoch += weighted_recall_batch
            total_weighted_fallout_epoch += weighted_fallout_batch
            total_weighted_iou_epoch += weighted_iou_batch
            total_weighted_f1_epoch += weighted_f1_batch

        mean_loss_epoch = total_loss_epoch / (idx_batch + 1)
        mean_bdte_epoch = total_bdte_epoch / (idx_batch + 1)
        mean_acc_epoch = total_acc_epoch / (idx_batch + 1)
        mean_precision_epoch = total_precision_epoch / (idx_batch + 1)
        mean_recall_epoch = total_recall_epoch / (idx_batch + 1)
        mean_fallout_epoch = total_fallout_epoch / (idx_batch + 1)
        mean_iou_epoch = total_iou_epoch / (idx_batch + 1)
        mean_f1_epoch = total_f1_epoch / (idx_batch + 1)
        mean_weighted_acc_epoch = total_weighted_acc_epoch / (idx_batch + 1)
        mean_weighted_precision_epoch = total_weighted_precision_epoch / (idx_batch + 1)
        mean_weighted_recall_epoch = total_weighted_recall_epoch / (idx_batch + 1)
        mean_weighted_fallout_epoch = total_weighted_fallout_epoch / (idx_batch + 1)
        mean_weighted_iou_epoch = total_weighted_iou_epoch / (idx_batch + 1)
        mean_weighted_f1_epoch = total_weighted_f1_epoch / (idx_batch + 1)

        if rank == 0 or is_distributed == False:
            print(f"\tLoss train: {mean_loss_epoch:.5f}", flush=True)
            print(f"\tMean BDTE train: {mean_bdte_epoch:.5f}", flush=True)
        
        metrics_values = [mean_loss_epoch, mean_bdte_epoch, mean_acc_epoch, mean_precision_epoch, mean_recall_epoch, mean_fallout_epoch, mean_iou_epoch, mean_f1_epoch,
                          mean_weighted_acc_epoch, mean_weighted_precision_epoch, mean_weighted_recall_epoch, mean_weighted_fallout_epoch, mean_weighted_iou_epoch,
                          mean_weighted_f1_epoch]

        for i in range(len(train_metrics_names)):
            name_metric = train_metrics_names[i]
            value = metrics_values[i]
            train_metrics[name_metric].append(value)

        if (rank == 0 or is_distributed == False) and ((epoch + 1) % 5) == 0:
            with open(f"{experiment_dir}/all_metrics_train.json", "w") as file:
                json.dump(train_metrics, file, indent=4)

        ##################################################################

        if (rank == 0 or is_distributed == False) and val_dataloader is not None:
            print('Validation...')
            
            if is_distributed:
                train_dataloader.sampler.set_epoch(epoch) 
            
            model.eval()

            total_loss_epoch = 0
            total_bdte_epoch = 0
            total_acc_epoch = 0
            total_precision_epoch = 0
            total_recall_epoch = 0
            total_fallout_epoch = 0
            total_iou_epoch = 0
            total_f1_epoch = 0
            total_weighted_acc_epoch = 0
            total_weighted_precision_epoch = 0
            total_weighted_recall_epoch = 0
            total_weighted_fallout_epoch = 0
            total_weighted_iou_epoch = 0
            total_weighted_f1_epoch = 0
            
            with torch.no_grad():
    
                for idx_batch, (input_batch, ground_truth_batch) in enumerate(val_dataloader):

                    output_batch = model(input_batch)
                    ground_truth_batch = ground_truth_batch.float()
                    loss = loss_function(output_batch, ground_truth_batch)
                    loss_batch = loss.detach().item()
                    total_loss_epoch += loss_batch
        
                    # IMPORTANTE! Como as saídas são em logits, é necessário binarizar com a sigmoide para calcular corretamente as métricas
                    output_batch = torch.sigmoid(output_batch)
                    output_batch = (output_batch > 0.5).to(torch.uint8)
        
                    # Métricas convencionais, sem expansão das bordas
                    bdte_batch, acc_batch, precision_batch, recall_batch, fallout_batch, iou_batch, f1_batch = binary_metrics(ground_truth_batch, output_batch, weighted_metrics=False)
                    total_bdte_epoch += bdte_batch
                    total_acc_epoch += acc_batch
                    total_precision_epoch += precision_batch
                    total_recall_epoch += recall_batch
                    total_fallout_epoch += fallout_batch
                    total_iou_epoch += iou_batch
                    total_f1_epoch += f1_batch
        
                    # Métricas ponderadas
                    _, weighted_acc_batch, weighted_precision_batch, weighted_recall_batch, weighted_fallout_batch, weighted_iou_batch, weighted_f1_batch = binary_metrics(ground_truth_batch, output_batch, weighted_metrics=True)
                    total_weighted_acc_epoch += weighted_acc_batch
                    total_weighted_precision_epoch += weighted_precision_batch
                    total_weighted_recall_epoch += weighted_recall_batch
                    total_weighted_fallout_epoch += weighted_fallout_batch
                    total_weighted_iou_epoch += weighted_iou_batch
                    total_weighted_f1_epoch += weighted_f1_batch

                mean_loss_epoch = total_loss_epoch / (idx_batch + 1)
                mean_bdte_epoch = total_bdte_epoch / (idx_batch + 1)
                mean_acc_epoch = total_acc_epoch / (idx_batch + 1)
                mean_precision_epoch = total_precision_epoch / (idx_batch + 1)
                mean_recall_epoch = total_recall_epoch / (idx_batch + 1)
                mean_fallout_epoch = total_fallout_epoch / (idx_batch + 1)
                mean_iou_epoch = total_iou_epoch / (idx_batch + 1)
                mean_f1_epoch = total_f1_epoch / (idx_batch + 1)
                mean_weighted_acc_epoch = total_weighted_acc_epoch / (idx_batch + 1)
                mean_weighted_precision_epoch = total_weighted_precision_epoch / (idx_batch + 1)
                mean_weighted_recall_epoch = total_weighted_recall_epoch / (idx_batch + 1)
                mean_weighted_fallout_epoch = total_weighted_fallout_epoch / (idx_batch + 1)
                mean_weighted_iou_epoch = total_weighted_iou_epoch / (idx_batch + 1)
                mean_weighted_f1_epoch = total_weighted_f1_epoch / (idx_batch + 1)

                if rank == 0 or is_distributed == False:
                    print(f"\tLoss validation: {mean_loss_epoch:.5f}", flush=True)
                    print(f"\tMean BDTE validation: {mean_bdte_epoch:.5f}", flush=True)
                
                metrics_values = [mean_loss_epoch, mean_bdte_epoch, mean_acc_epoch, mean_precision_epoch, mean_recall_epoch, mean_fallout_epoch, mean_iou_epoch, mean_f1_epoch,
                                  mean_weighted_acc_epoch, mean_weighted_precision_epoch, mean_weighted_recall_epoch, mean_weighted_fallout_epoch, mean_weighted_iou_epoch,
                                  mean_weighted_f1_epoch]
        
                for i in range(len(val_metrics_names)):
                    name_metric = val_metrics_names[i]
                    value = metrics_values[i]
                    val_metrics[name_metric].append(value)
    
                if (epoch + 1) % 5 == 0:
                    with open(f"{experiment_dir}/all_metrics_validation.json", "w") as file:
                        json.dump(val_metrics, file, indent=4)

                ############# salvamento do melhor modelo ###############

                if (rank == 0 or is_distributed == False) and (epoch >= 99) and (mean_bdte_epoch <= min_bdte_validation):
                    epoch_best_model = epoch
                    min_bdte_validation = mean_bdte_epoch
                    # Salvando modelo e otimizador
                    torch.save({"model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict()},
                               f"{experiment_dir}/complete_model.pth")

        if rank == 0 or is_distributed == False:
            print(f'Time: {time.time() - start_epoch:,.2f} sec', flush=True)

        if epoch == epoch_best_model + patience:
            print('----------------------------------------------', flush=True)
            print('TRAINING INTERRUPTED EARLIER, MODEL STOPPED IMPROVING VALIDATION.')
            break
     
    if rank == 0 or is_distributed == False:
        print('----------------------------------------------', flush=True)
        print(f'Min validation BDTE: {min_bdte_validation:.5f}. Epoch: {epoch_best_model + 1}', flush=True)
        print('----------------------------------------------', flush=True)
        print(f'Total time: {(time.time() - start_total)/60:,.1f} min')
        print('----------------------------------------------', flush=True)
