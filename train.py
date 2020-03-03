import numpy as np
from keras.datasets import mnist
from layers.conv import Conv2D
from layers.max_pool import MaxPooling2D
from layers.softmax import Softmax

print("Loading MNIST")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_images = x_train[:1500]
train_labels = y_train[:1500]
test_images = x_test[:1500]
test_labels = y_test[:1500]

NUM_FILTERS = 12
FILTER_SIZE = 3


conv = Conv2D(NUM_FILTERS, FILTER_SIZE)
pool = MaxPooling2D(2)
softmax = Softmax(13*13*12, 10)
print("CNN layers created")

def forward_cnn(image, label):
    out = conv.forward((image/255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    cross_ent_loss = -np.log(out[label])
    accuracy = 1 if np.argmax(out) == label else 0
    return out, cross_ent_loss, accuracy

def train(image, label, learning_rate=0.005):
    out, loss, acc = forward_cnn(image, label)
    gradient = np.zeros(10)
    gradient[label] = -1/out[label]

    grad_back = softmax.backward(gradient, learning_rate)
    grad_back = pool.backward(grad_back)
    grad_back = conv.backward(grad_back, learning_rate)
    return loss, acc

epochs = 5
for epoch in range(epochs):
    print("Epoch :{}".format(epoch))
    print('**Train**')
    shuffle_data = np.random.permutation(len(train_images))
    train_images = train_images[shuffle_data]
    train_labels = train_labels[shuffle_data]

    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        l1, accu = train(im, label)
        loss += l1
        num_correct += accu
        
    num_train = len(train_images)
    print('Train Loss:', loss/num_train)
    print('Train Accuracy:', num_correct/num_train)

    print('**Test**')
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, l1, accu = forward_cnn(im, label)
        loss += l1
        num_correct += accu
    
    num_tests = len(test_images)
    print('Test Loss:', loss/num_tests)
    print('Test Accuracy:', num_correct/num_tests)