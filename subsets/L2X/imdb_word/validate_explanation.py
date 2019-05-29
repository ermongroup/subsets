"""
Compute the accuracy with selected words by using L2X.
"""
import os

import numpy as np
import click

from subsets.L2X.imdb_word.explain import create_original_model


@click.command()
@click.option('--task', required=True, help="'l2x' or 'subsets'")
@click.option('--tau', required=True, type=float, help="temperature used to train")
def validate(task, tau):
    print('Loading dataset...')
    x_val_selected = np.load(f'data/x_val-{task}-{tau}.npy')
    pred_val = np.load('data/pred_val.npy')
    print('Creating model...')
    model = create_original_model()

    weights_name = [
        i for i in os.listdir('./models') if i.startswith('original')][0]
    model.load_weights('./models/' + weights_name, by_name=True)
    new_pred_val = model.predict(
        x_val_selected, verbose=1, batch_size=1000)
    val_acc = np.mean(
        np.argmax(pred_val, axis=-1) == np.argmax(new_pred_val, axis=-1))
    print(f"The validation accuracy of {task}-{tau}"
          f"with selected 10 words is {val_acc}+-{1.96*np.sqrt(val_acc*(1-val_acc)/pred_val.shape[0])}.")
    np.save(f'data/pred_val-{task}-{tau}.npy', new_pred_val)


if __name__ == '__main__':
    validate()
