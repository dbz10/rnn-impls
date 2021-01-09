"""
Vanilla char level RNN implementation. Obviously inspired by
https://karpathy.github.io/2015/05/21/rnn-effectiveness/ + https://gist.github.com/karpathy/d4dee566867f8291f086
author: Daniel Ben-Zion
2021/01/01
"""

import numpy as np
from copy import deepcopy
import joblib


class RNN:
    def __init__(self, vocabulary, hidden_size=10, seq_length=25, learning_rate=1e-2, ):
        self.char_encoder = vocabulary['encoder']
        self.char_decoder = vocabulary['decoder']
        self.vocab_size = vocabulary['vocab_size']
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # initialize weights to small random numbers
        # RNN's can have a tendency for instability so initializing to very small
        # numbers is advised.
        self.h = np.zeros((hidden_size, 1))
        self.Whh = np.random.normal(0, 0.01, size=(hidden_size, hidden_size))
        self.Wyh = np.random.normal(0, 0.01, size=(self.vocab_size, hidden_size))
        # self.Wyh = np.zeros((self.vocab_size, hidden_size))
        self.Whx = np.random.normal(0, 0.01, size=(hidden_size, self.vocab_size))
        self.by = np.random.normal(0, 0.01, size=(self.vocab_size, 1))
        # self.by = np.zeros((self.vocab_size, 1))
        self.bh = np.random.normal(0, 0.01, size=(self.hidden_size, 1))

    def read_prompt(self, inputs):
        """
        consumes a series in input characters to "prime" the state vector h.
        :param inputs:
        :return:
        """
        # initialize h to the current h state
        h = np.zeros((self.hidden_size, 1))
        # forward prop
        for char in inputs:
            x = np.zeros((self.vocab_size, 1))  # (vocab_size, 1)  vector
            x[self.char_encoder[char]] = 1
            h = np.tanh(np.dot(self.Whx, x) + np.dot(self.Whh, h) + self.bh)  # (hidden_size, 1) vector

        return h

    def fast_loss(self, inputs, targets):
        xs, hs, ys, probas = {}, {}, {}, {}
        losses = np.zeros(len(inputs))
        hs[-1] = np.zeros_like(self.h)
        # forward prop
        for t, (xin, target) in enumerate(zip(inputs, targets)):
            xs[t] = np.zeros((self.vocab_size, 1))  # (vocab_size, 1)  vector
            xs[t][self.char_encoder[xin]] = 1
            hs[t] = np.tanh(np.dot(self.Whx, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # (hidden_size, 1) vector
            ys[t] = np.dot(self.Wyh, hs[t]) + self.by  # (vocab_size, 1) vector
            probas[t] = np.exp(ys[t]) / np.exp(ys[t]).sum()  # (vocab_size, 1) vector
            target_idx = self.char_encoder[target]
            losses[t] = -np.log(probas[t][target_idx, 0])

        return losses.mean()

    def loss_function(self, inputs, targets):
        """
        :param inputs:
        :param targets:
        :return: L, dWhh, dWyh, dWhx, dbh, dby, h_final
        """
        xs, hs, ys, probas = {}, {}, {}, {}
        losses = np.zeros(len(inputs))

        # initialize h to the current h state
        hs[-1] = self.h.copy()
        # forward prop
        for t, (xin, target) in enumerate(zip(inputs, targets)):
            xs[t] = np.zeros((self.vocab_size, 1))  # (vocab_size, 1)  vector
            xs[t][self.char_encoder[xin]] = 1
            hs[t] = np.tanh(np.dot(self.Whx, xs[t]) + np.dot(self.Whh, hs[t - 1]) + self.bh)  # (hidden_size, 1) vector
            ys[t] = np.dot(self.Wyh, hs[t]) + self.by  # (vocab_size, 1) vector
            probas[t] = np.exp(ys[t]) / np.exp(ys[t]).sum()  # (vocab_size, 1) vector
            target_idx = self.char_encoder[target]
            losses[t] = -np.log(probas[t][target_idx, 0])

        # back prop to get the gradients
        dWyh = np.zeros_like(self.Wyh)
        dWhh = np.zeros_like(self.Whh)
        dWhx = np.zeros_like(self.Whx)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        # there is actually one slightly subtle thing about backprop for Whh related
        # to which time index is used for h, since Whh connects h[t] to h[t+1]
        # so at each step we are dealing with two h's

        delta_hs = {}
        delta_ys = {}

        delta_hs[len(xs)] = np.zeros_like(self.h)  # this is what karpathy intializes as dhnext.
        # I suppose the point is that because it's the last h it actually doesn't affect the loss at all.
        # delta being defined as dE/dz gives delta_hlast = 0. So here we are initializing the delta h
        # after the last h that actually contributes to y to be zero.
        # It's a bit confusing I guess. In fact that last h doesn't even exist because
        # there's no x[t+1] to form it. ¯\_(ツ)_/¯

        for t, (xin, target) in reversed(list(enumerate(zip(inputs, targets)))):
            target_idx = self.char_encoder[target]
            delta_ys[t] = probas[t].copy()
            delta_ys[t][target_idx] -= 1

            # backprop through the tanh
            # split into two lines just for pep8
            delta_hs[t] = np.dot(self.Wyh.T, delta_ys[t]) + np.dot(self.Whh.T, delta_hs[t + 1])
            delta_hs[t] = delta_hs[t] * (1 - hs[t] * hs[t])

            dby += delta_ys[t]
            dbh += delta_hs[t]
            dWyh += np.dot(delta_ys[t], hs[t].T)
            dWhh += np.dot(delta_hs[t + 1], hs[t].T)  # note the t and t+1!
            dWhx += np.dot(delta_hs[t], xs[t].T)

        # apparently RNNs are particularly susceptible to exploding gradients, so
        for grad in [dWyh, dWhh, dWhx, dby, dbh]:
            np.clip(grad, -3, 3, out=grad)

        # note: we don't mutate the self.h state during this function. Instead we return
        # it and the calling method will update h. This is cleaner.
        output = {
            'loss': losses.mean(),
            'dWyh': dWyh,
            'dWhh': dWhh,
            'dWhx': dWhx,
            'dbh': dbh,
            'dby': dby,
            'h_new': hs[len(inputs) - 1]  # hs is a dictionary not a list, so hs[-1] is not correct
        }
        return output

    def train(self, X_train, y_train, X_val, y_val,
              n_iterations=100,
              early_stopping_rounds=None,
              monitoring=None,
              update="sgd"):
        # initialize loss tracking
        training_loss = []
        validation_loss = []
        training_loss.append(self.fast_loss(X_train, y_train))
        validation_loss.append(self.fast_loss(X_val, y_val))

        print("Intial train/val losses:", training_loss[0], validation_loss[0])

        # initialize early stopping tracking
        consecutive_val_loss_plateau = 0

        # initialize memory for certain update types
        self.initialize_memory(update)

        for iteration in range(int(n_iterations)):
            position = 0
            training_loss_accumulator = []
            # reset internal state
            self.h = np.zeros((self.hidden_size, 1))
            while position + self.seq_length < len(X_train):
                inputs = X_train[position:position + self.seq_length]
                targets = y_train[position:position + self.seq_length]
                gradients = self.loss_function(inputs, targets)
                self.h = gradients['h_new']
                self.weight_update(gradients, update)
                training_loss_accumulator.append(gradients['loss'])
                position += self.seq_length

            training_loss.append(np.mean(training_loss_accumulator))
            validation_loss.append(self.fast_loss(X_val, y_val))

            if monitoring == "text" and iteration % 1 == 0:
                print("-" * 40)
                print("Iteration:", iteration)
                print("Training Loss:", training_loss[-1])
                print("Validation Loss:", validation_loss[-1])

            if validation_loss[-1] - validation_loss[-2] > -3e-4:
                consecutive_val_loss_plateau += 1
            else:
                consecutive_val_loss_plateau = max(consecutive_val_loss_plateau - 1, 0)

            # checkpoint the model every so often. \
            if iteration % 10 == 0:
                joblib.dump(self, f"model_checkpoints/model_{iteration}_{validation_loss[-1]}.joblib")

            if early_stopping_rounds is not None and consecutive_val_loss_plateau > early_stopping_rounds:
                return training_loss, validation_loss

        return training_loss, validation_loss

    def weight_update(self, gradients, update):
        # vanilla SGD
        if update == "sgd":
            self.Whh -= self.learning_rate * gradients['dWhh']
            self.Wyh -= self.learning_rate * gradients['dWyh']
            self.Whx -= self.learning_rate * gradients['dWhx']
            self.bh -= self.learning_rate * gradients['dbh']
            self.by -= self.learning_rate * gradients['dby']

        # adagrad update from karpathy, i don't pretend to understand this (yet)
        if update == 'adagrad':
            for (weight, grad, mem) in zip(
                    [self.Wyh, self.Whh, self.Whx, self.bh, self.by],
                    [gradients['dWyh'], gradients['dWhh'], gradients['dWhx'], gradients['dbh'], gradients['dby']],
                    [self.mWyh, self.mWhh, self.mWhx, self.mbh, self.mby]
            ):
                mem += grad * grad
                weight -= self.learning_rate * grad / np.sqrt(mem + 1e-8)

    def initialize_memory(self, update):
        # adagrad update from karpathy
        if update == 'adagrad':
            self.mWhh = np.zeros_like(self.Whh)
            self.mWhx = np.zeros_like(self.Whx)
            self.mWyh = np.zeros_like(self.Wyh)
            self.mbh = np.zeros_like(self.bh)
            self.mby = np.zeros_like(self.by)

    def gradient_checking(self, inputs, targets):
        """
        Compute derivative of cost function with respect to weights "manually".
        This is quite slow to do so adds a lot to the training time if enabled.
        :param inputs:
        :param targets:
        :return:
        """
        # idk if this is kosher but create a mock so we're not mutating the state
        # of the object itself while we're doing this
        mock = deepcopy(self)

        epsilon = 1e-5

        gradient_absolute_errors = []
        actual_gradients = []
        # compute the gradient via backprop
        grad = mock.loss_function(inputs, targets)
        for (parm, weight) in zip(
                ['dWyh', 'dWhh', 'dWhx', 'dbh', 'dby'],
                [mock.Wyh, mock.Whh, mock.Whx, mock.bh, mock.by]
        ):
            analytical_gradient = grad[parm]
            for idx in range(len(weight)):
                w0 = weight.flat[idx]
                weight.flat[idx] = w0 + epsilon
                # remember loss is mean loss, but gradient is wrt total loss
                l_plus = mock.loss_function(inputs, targets)['loss'] * len(inputs)
                weight.flat[idx] = w0 - epsilon
                l_minus = mock.loss_function(inputs, targets)['loss'] * len(inputs)
                weight.flat[idx] = w0

                numerical_gradient = (l_plus - l_minus) / (2 * epsilon)
                clipped_gradient = np.clip(numerical_gradient, -3, 3)
                gradient_absolute_error = np.abs(clipped_gradient - analytical_gradient.flat[idx])
                gradient_absolute_errors.append(gradient_absolute_error)
                actual_gradients.append(numerical_gradient)

        print("Mean absolute error of gradient terms:", np.array(gradient_absolute_errors).mean())
        print("p99 of clipped gradient errors:", np.quantile(gradient_absolute_errors, 0.99))

    def sample(self, seed, beta, n):
        """
        sample from the RNN to generate a sequence of outputs after priming with a seed
        """
        # prime the hidden state by reading the prompt
        mock = deepcopy(self)
        mock.h = mock.read_prompt(seed)
        output = [seed[-1]]
        for i in range(n):
            xin = np.zeros((mock.vocab_size, 1))
            xin[mock.char_encoder[output[-1]]] = 1
            mock.h = np.tanh(np.dot(mock.Whx, xin) + np.dot(mock.Whh, mock.h) + mock.bh)  # (hidden_size, 1) vector
            y = np.dot(mock.Wyh, mock.h) + mock.by  # (vocab_size, 1) vector
            probas = np.exp(beta * y) / np.exp(beta * y).sum()  # (vocab_size, 1) vector
            output_idx = np.random.choice(range(mock.vocab_size), p=probas.ravel())
            output_char = mock.char_decoder[output_idx]
            output.append(output_char)

        return output
