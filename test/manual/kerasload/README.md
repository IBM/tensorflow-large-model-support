# Keras save / load testing

## Background
Keras' save model functionality does not save the nodes that TFLMS adds to
the graph. It saves layer definitions. The save model functionality also
saves the weights, the optimizer weights, and the optimizer configuration/state.

During load, the weights are set back on the instantiated model using a
Session. This finalizes the model and prevents TFLMS from adding nodes after
the load. An API change was added to TensorFlow Keras in PowerAI to allow
the load_model method to take a list of Keras callbacks. This same change
has been proposed upstream. The change calls the set_model method on the
callbacks after the load routine instantiates the model but before the
weights and optimizer weights and settings are applied.

In addition to the save model method, Keras provides a method to save just
the weights and can save them in two different formats: the TensorFlow
checkpoint format or the Keras .h5 format.

This has ramifications to Keras save and load and user scripts that are
saving and loading models or weights.

When these save and load options are combined with the various use cases it
provides many paths to test for LMS. For instance:

train > save > load_model > resume training
train > save > inference / evaluate
train > save weights > instantiate model with code > inference / evaluate
train > save weights > instantiate model with code > resume training

In practice with the test files, we have seen that only the load_model is
finalizing the graph. Loading the weights on a model created via code does
not prevent TFLMS from modifying it.

## Testing files
The files in this directory are meant to be run to test the various
scenarios listed above as well as other scenarios that could be imagined.

The files are not guaranteed to be "complete" and testing using them will
involve modifying the save / load sections to achieve different test combinations.

In general, the train_save .py should be run first to do some training and
produce the saved file. The corresponding load_train or load_<somethingelse>
file is then run to test the ability for LMS to modify the graph.
