# Number Recognition
An artificial neural network that tries to classify handwritten digits.

Uses my neural network library [Neurolib](https://github.com/bbssamuray/Neurolib) that is written from scratch in c++.

It uses the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset for training and testing.

Depending on the layer count and size it reaches about %97.5~ success rate on the test dataset after training.


# Building

You can build this project just by doing:
```sh
  git clone https://github.com/bbssamuray/NumberRecognition
  cd NumberRecognition
  make
```
You can then run ``numberRecognition.exe``.

# Usage

```sh
  ./numberRecognition.exe [modelName]
```
If the given model exists, it will try to guess the number written in ``number.bmp``.

``number.bmp`` **must** be a 28x28 pixel bmp file, within it there should be a digit [0-9] written with white color on a black background.


If no model name is specified, it will just use ``numberModel.o``.

# Training and Testing

If you give the binary a model name that doesn't exist, it will start training if the necessary training files are present.

```sh
  ./numberRecognition.exe exampleModel.o
```
Snippet above will start training a model called ``exampleModel.o``.

For training to start, you need to have ``train-images-idx3-ubyte`` and ``train-labels-idx1-ubyte`` in the current directory.

You can acquire those files from [Here](http://yann.lecun.com/exdb/mnist/).

Testing is the exact same, but is done when you give it existing models. Testing requires the ``t10k-images-idx3-ubyte`` and ``t10k-labels-idx1-ubyte`` files, which you can get from the [same place](http://yann.lecun.com/exdb/mnist/).

Some archive tools change files names when decompressing. You might want to double check if you are getting any "not found" errors.
