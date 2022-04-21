import argparse

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def evaluate_split(labels):
    labels = labels.cpu()
    positives = (labels == 1).sum()
    negatives = (labels == 0).sum()

    assert positives + negatives == labels.shape[0]

    if positives > negatives:
        predictions = torch.ones_like(labels).cpu()
    else:
        predictions = torch.zeros_like(labels).cpu()

    print(f'positive:               {100 * float(positives) / labels.shape[0]:.1f}')
    print(f'accuracy:               {100 * accuracy_score(labels, predictions):.1f}')
    print(f'balanced accuracy:      {100 * balanced_accuracy_score(labels, predictions):.1f}')
    print(f'f1 score:               {f1_score(labels, predictions)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--load', action='store_true')
    parser.add_argument('--transfer', action='store_true')
    parser.add_argument('--protected_att', type=str, default=None)
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--quantiles', action='store_true')
    args = parser.parse_args()

    dataset = getattr(__import__('datasets'), args.dataset.capitalize() + 'Dataset')('train', args)

    print('Train')
    evaluate_split(dataset.y_train)
    print(f'sensitive:              {100 * dataset.protected_train.float().mean():.1f}')
    print(f'positive/non-sensitive: {100 * dataset.y_train[dataset.protected_train == 0].float().mean():.1f}')
    print(f'positive/sensitive:     {100 * dataset.y_train[dataset.protected_train == 1].float().mean():.1f}')
    print(f'Samples:                {dataset.y_train.shape[0]}')
    print()

    print('Validation')
    evaluate_split(dataset.y_val)
    print(f'sensitive:              {100 * dataset.protected_val.float().mean():.1f}')
    print(f'positive/non-sensitive: {100 * dataset.y_val[dataset.protected_val == 0].float().mean():.1f}')
    print(f'positive/sensitive:     {100 * dataset.y_val[dataset.protected_val == 1].float().mean():.1f}')
    print(f'Samples:                {dataset.y_val.shape[0]}')
    print()

    print('Test')
    evaluate_split(dataset.y_test)
    print(f'sensitive:              {100 * dataset.protected_test.float().mean():.1f}')
    print(f'positive/non-sensitive: {100 * dataset.y_test[dataset.protected_test == 0].float().mean():.1f}')
    print(f'positive/sensitive:     {100 * dataset.y_test[dataset.protected_test == 1].float().mean():.1f}')
    print(f'Samples:                {dataset.y_test.shape[0]}')
    print()
