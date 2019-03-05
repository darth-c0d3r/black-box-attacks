### Black-Box Attacks on Neural Networks

Conv1
conv = [1, 4, 8, 16]
fc = [128, 64]
EPOCHS = 20
BATCH_SIZE = 1000
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
Eval Accuracy = 98.41 %

Conv2
conv = [1, 4, 8, 16, 32]
fc = []
EPOCHS = 20
BATCH_SIZE = 1000
optimizer = optim.Adagrad(model.parameters(), lr=0.01)
Eval Accuracy = 98.63 %
