import numpy as np
import pickle


def get_selected_words(x_single, score, id_to_word, k):
    selected = np.argsort(score)[-k:]
    selected_k_hot = np.zeros(400)
    selected_k_hot[selected] = 1.0

    x_selected = (x_single * selected_k_hot).astype(int)
    return x_selected


def create_dataset_from_score(x, scores, k, task, tau):
    with open('data/id_to_word.pkl', 'rb') as f:
        id_to_word = pickle.load(f)
    new_data = []
    for i, x_single in enumerate(x):
        x_selected = get_selected_words(x_single, scores[i], id_to_word, k)
        new_data.append(x_selected)

    np.save(f'data/x_val-{task}-{tau}.npy', np.array(new_data))


def calculate_acc(pred, y):
    return np.mean(np.argmax(pred, axis=1) == np.argmax(y, axis=1))
