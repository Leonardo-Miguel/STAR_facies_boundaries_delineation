from utils.Metrics import binary_metrics

import torch
import numpy as np
import time
from sklearn.metrics import r2_score

def predict_boundaries(model, test_dataloader, samples_partition, section_type, volume_shape, output_dir, output_name, save_prediction, rank):

    start_total = time.time()

    output_volume = np.zeros(volume_shape, dtype=np.float32)

    total_bdte = 0
    total_acc = 0
    total_precision = 0
    total_recall = 0
    total_fallout = 0
    total_iou = 0
    total_f1 = 0
    total_weighted_acc = 0
    total_weighted_precision = 0
    total_weighted_recall = 0
    total_weighted_fallout = 0
    total_weighted_iou = 0
    total_weighted_f1 = 0

    model.eval()
    print('Testing model', flush=True)

    n_valid_batches = len(test_dataloader)

    original_position_sample = 0 
    
    with torch.no_grad():

        for idx_batch, (input_batch, ground_truth_batch) in enumerate(test_dataloader):

            print(f'Batch test {idx_batch + 1}/{len(test_dataloader)}', flush=True)
            
            output_batch = model(input_batch)

            # IMPORTANTE! Como as saídas são em logits, é necessário binarizar com a sigmoide para calcular corretamente as métricas
            output_batch = torch.sigmoid(output_batch)
            output_batch = (output_batch > 0.5).to(torch.uint8)
            output_batch = output_batch.cpu().numpy()
            ground_truth_batch = ground_truth_batch.float() # É NECESSÁRIO CONVERTER PARA FLOAT, para o cálculo das distâncias para as métricas ponderadas. Caso contrário ele arredonda, se for int
            ground_truth_batch = ground_truth_batch.cpu().numpy()

            evaluated_samples = []

            for batch_sample in range(output_batch.shape[0]):

                output_sample = output_batch[batch_sample, :, :, :]

                if section_type == 'inline':
                    output_volume[original_position_sample, :, :] = output_sample
                if section_type == 'crossline':
                    output_volume[:, original_position_sample, :] = output_sample
                if section_type == 'timeslice':
                    output_volume[:, :, original_position_sample] = output_sample

                # verificação de quais amostras serão consideradas para cálculo das métricas
                # já que todo o volume passa pelo teste, mas as amostras consideradas são só as que realmente pertencem à partição
                if original_position_sample in samples_partition:
                    evaluated_samples.append(batch_sample)

                original_position_sample += 1

            ground_truth_batch = ground_truth_batch[evaluated_samples, :, :, :]
            output_batch = output_batch[evaluated_samples, :, :, :]

            # para que os batches que não têm amostras de teste sejam desconsiderados no cálculo das métricas e das médias ao final
            if evaluated_samples:
                # Métricas convencionais, sem expansão das bordas
                bdte_batch, acc_batch, precision_batch, recall_batch, fallout_batch, iou_batch, f1_batch = binary_metrics(ground_truth_batch, output_batch, weighted_metrics=False)
                total_bdte += bdte_batch
                total_acc += acc_batch
                total_precision += precision_batch
                total_recall += recall_batch
                total_fallout += fallout_batch
                total_iou += iou_batch
                total_f1 += f1_batch
    
                # Métricas ponderadas
                bdte_batch, weighted_acc_batch, weighted_precision_batch, weighted_recall_batch, weighted_fallout_batch, weighted_iou_batch, weighted_f1_batch = binary_metrics(ground_truth_batch, output_batch, weighted_metrics=True)
                total_weighted_acc += weighted_acc_batch
                total_weighted_precision += weighted_precision_batch
                total_weighted_recall += weighted_recall_batch
                total_weighted_fallout += weighted_fallout_batch
                total_weighted_iou += weighted_iou_batch
                total_weighted_f1 += weighted_f1_batch
            else:
                n_valid_batches -= 1
                            
            del input_batch, output_batch
            torch.cuda.empty_cache()

    mean_bdte = total_bdte / n_valid_batches
    mean_acc = total_acc / n_valid_batches
    mean_precision = total_precision / n_valid_batches
    mean_recall = total_recall / n_valid_batches
    mean_fallout = total_fallout / n_valid_batches
    mean_iou = total_iou / n_valid_batches
    mean_f1 = total_f1 / n_valid_batches
    mean_weighted_acc = total_weighted_acc / n_valid_batches
    mean_weighted_precision = total_weighted_precision / n_valid_batches
    mean_weighted_recall = total_weighted_recall / n_valid_batches
    mean_weighted_fallout = total_weighted_fallout / n_valid_batches
    mean_weighted_iou = total_weighted_iou / n_valid_batches
    mean_weighted_f1 = total_weighted_f1 / n_valid_batches
    

    print(f'''
mean_bdte: {mean_bdte}
mean_iou: {mean_iou}
mean_f1: {mean_f1}
mean_precision: {mean_precision}
mean_recall: {mean_recall}
mean_fallout: {mean_fallout}
mean_acc: {mean_acc}
mean_weighted_iou: {mean_weighted_iou}
mean_weighted_f1: {mean_weighted_f1}
mean_weighted_precision: {mean_weighted_precision}
mean_weighted_recall: {mean_weighted_recall}
mean_weighted_fallout: {mean_weighted_fallout}
mean_weighted_acc: {mean_weighted_acc}\n''')

    print(f'''{output_name}
{round(mean_bdte, 3)}
{round(mean_weighted_iou, 3)}
{round(mean_iou, 3)}
{round(mean_weighted_f1, 3)}
{round(mean_f1, 3)}
{round(mean_weighted_precision, 3)}
{round(mean_precision, 3)}
{round(mean_weighted_recall, 3)}
{round(mean_recall, 3)}
{round(mean_weighted_acc, 3)}
{round(mean_acc, 3)}\n'''.replace('\n', '\t').replace('.', ','))

    if save_prediction:
        np.save(f'{output_dir}/{output_name}_boundaries_prediction.npy', output_volume)
        print(f'{output_dir}/{output_name}_boundaries_prediction.npy saved', flush=True)
    
    print('Ending program.', flush=True)
    print(f'Total time: {(time.time() - start_total)/60:,.1f} min', flush=True)
