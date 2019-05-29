"""
Compute the accuracy with selected sentences by using L2X.
"""
import numpy as np
import click

from subsets.L2X.imdb_sent.make_data import load_data
from subsets.L2X.imdb_sent.explain import create_original_model


@click.command()
@click.option('--task', required=True, help="'l2x' or 'subsets'")
@click.option('--tau', required=True, type=float, help="temperature used to train")
def validate(task, tau):
    print('Loading dataset...')
    dataset = load_data()
    word_index = dataset['word_index']
    x_val = np.load(f'data/x_val-{task}-{tau}.npy')
    pred_val = np.load('data/pred_val.npy')

    print('Creating model...')
    model = create_original_model(word_index)
    model.load_weights('./models/original.hdf5', by_name=True)

    print('Making prediction with selected sentences...')
    new_pred_val = model.predict(x_val, verbose=1, batch_size=1000)
    val_acc = (np.mean(np.argmax(new_pred_val, axis=-1)
                       == np.argmax(pred_val, axis=-1)))

    print(f"The validation accuracy of {task}-{tau}"
          f"is {val_acc}+-{1.96*np.sqrt(val_acc*(1-val_acc)/pred_val.shape[0])}.")
    np.save(f'data/pred_val-{task}-{tau}.npy', new_pred_val)


if __name__ == '__main__':
    validate()
