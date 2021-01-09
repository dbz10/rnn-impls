import argparse
import joblib
from dio import load_data
import preprocessing
from vanilla_rnn import RNN
import numpy as np

# hyperparameters
hidden_size = 250
seq_length = 50
learning_rate = 3e-4
early_stopping_rounds = 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest="input_file", type=str, help="Input file containing training sequence(s).")
    parser.add_argument(dest="prediction_prompt", type=str, help="Prompt for making a prediction.")
    args = parser.parse_args()

    print("Training on file:", args.input_file)

    data = load_data(args.input_file)
    vocabulary = preprocessing.create_vocabulary(data)

    # Split into train test and validation set while preserving individual lines intact.
    input_lines, test_lines = preprocessing.train_test_split(data, test_size=0.1)
    train_lines, val_lines = preprocessing.train_test_split(input_lines, test_size=0.1)

    X_train, y_train = preprocessing.text_to_input_and_target(train_lines)
    X_test, y_test = preprocessing.text_to_input_and_target(test_lines)
    X_val, y_val = preprocessing.text_to_input_and_target(val_lines)

    print("Number of training examples:", len(X_train))
    count_parameters = (hidden_size * hidden_size  # Whh
                        + hidden_size * vocabulary['vocab_size']  # Whx
                        + hidden_size * vocabulary['vocab_size']  # Wyh
                        + hidden_size  # bh
                        + vocabulary['vocab_size']  # by
                        )

    print("Number of model parameters:", count_parameters)
    print("Expected init loss:", np.log(vocabulary['vocab_size']))

    network = RNN(
        vocabulary,
        hidden_size=hidden_size,
        seq_length=seq_length,
        learning_rate=learning_rate,
    )

    network.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_iterations=1e3,
        early_stopping_rounds=early_stopping_rounds,
        monitoring="text",
        update='sgd'
    )

    # evaluate the final loss on the training and validation set
    print('-' * 40)
    print("Final loss on train set:", network.fast_loss(X_train, y_train))
    print("Final loss on validation set:", network.fast_loss(X_val, y_val))

    # evaluate model loss on the test set
    print("Loss on test set:", network.fast_loss(X_test, y_test))
    print('-' * 40)
    print("\n")

    # save the trained network
    joblib.dump(network, "model.joblib")

    # Generate a prediction on a sample input.
    prompt = load_data(args.prediction_prompt)
    seed, _ = preprocessing.text_to_input_and_target(prompt)
    output = network.sample(seed, 1, 200)
    print("Prediction based on input prompt:")
    print(prompt)
    print("-" * 40)
    print(''.join(output))
    print("-" * 40)


if __name__ == '__main__':
    main()
