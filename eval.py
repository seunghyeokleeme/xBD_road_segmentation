#!/usr/bin/env python
import os
import argparse
from PIL import Image
import numpy as np
import json

def compute_localization_metrics(result_dir):
    """
    Computes various metrics for road localization (detection) evaluation
    by reading 'test_mask' and 'test_pred' files from the 'png' folder within
    the given result directory.

    Pixels with values greater than 0 are considered as road (1),
    while 0 represents the background (0) to create binary masks.
    True Positives (TP), False Positives (FP), False Negatives (FN),
    and True Negatives (TN) are calculated at the pixel level.
    Subsequently, precision, recall, F1 score, accuracy, and IoU are computed.

    Parameters:
        result_dir (str): Path to the results directory.

    Returns:
        metrics (dict): A dictionary containing the computed metrics:
            'precision': ...,
            'recall': ...,
            'f1': ...,
            'accuracy': ...,
            'iou': ...,
            'TP': ...,
            'FP': ...,
            'FN': ...,
            'TN': ...
    """
    png_dir = os.path.join(result_dir, 'png')
    
    # Generate and sort file lists for labels and predictions
    lst_labels = sorted([f for f in os.listdir(png_dir) if f.startswith('test_mask')])
    lst_preds = sorted([f for f in os.listdir(png_dir) if f.startswith('test_pred')])
    
    # Optional: You can uncomment these print statements for debugging or verification
    # print(f"Number of labels: {len(lst_labels)}")
    # print(f"Number of preds: {len(lst_preds)}")
    
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    
    for label_file, pred_file in zip(lst_labels, lst_preds):
        # Open images with PIL and convert to grayscale NumPy arrays
        label = np.array(Image.open(os.path.join(png_dir, label_file)).convert('L'))
        pred = np.array(Image.open(os.path.join(png_dir, pred_file)).convert('L'))
        
        # Binarize: values > 0 are road (1), 0 is background (0)
        label_bin = (label > 0).astype(np.uint8)
        pred_bin = (pred > 0).astype(np.uint8)
        
        # Calculate TP, FP, FN, TN for the current image
        TP = ((pred_bin == 1) & (label_bin == 1)).sum()
        FN = ((pred_bin == 0) & (label_bin == 1)).sum()
        FP = ((pred_bin == 1) & (label_bin == 0)).sum()
        TN = ((pred_bin == 0) & (label_bin == 0)).sum()
        
        # Accumulate totals
        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN
    
    # Calculate metrics, handling division by zero
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (total_TP + total_TN) / (total_TP + total_FP + total_FN + total_TN) if (total_TP + total_FP + total_FN + total_TN) > 0 else 0
    iou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0
    
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "iou": iou,
        "TP": int(total_TP),
        "FP": int(total_FP),
        "FN": int(total_FN),
        "TN": int(total_TN)
    }
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compute localization (road detection) metrics for semantic segmentation results'
    )
    parser.add_argument('--result_dir', type=str, default='./results_v1',
                        help='Path to results directory (default: ./results_v1)')
    parser.add_argument('--out_fp', type=str, default='localization_metrics.json',
                        help='Output JSON file path (default: localization_metrics.json)')
    args = parser.parse_args()
    
    metrics = compute_localization_metrics(args.result_dir)
    
    print("---")
    print("Localization Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"IoU      : {metrics['iou']:.4f}")
    print(f"TP: {metrics['TP']}, FP: {metrics['FP']}, FN: {metrics['FN']}, TN: {metrics['TN']}")
    print("---")
    
    with open(args.out_fp, 'w') as f:
        json.dump(metrics, f) # Use indent for pretty printing JSON
    print(f"Metrics successfully saved to {args.out_fp}")