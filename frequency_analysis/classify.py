from libs.FrequencySolver import FrequencySolver
from absl import app, flags, logging
from absl.flags import FLAGS
import pickle

# Choose the number of input images from each dataset, default is set to 1000
flags.DEFINE_integer('num_iter', 1000, 'Number of images that will be used form EACH dataset, e.g. if set to 500, '
                                       'there will be 500 fake images and 500 real images used')
flags.DEFINE_integer('features', 300, 'Number of features used for training')
# Indicate path to training/test set
flags.DEFINE_string('data_path', None, 'Path to dataset for computing input features')

# flags.DEFINE_bool('compute_data', True, 'If training data is not saved, compute new training data using given paths')
# flags.DEFINE_string('saved_data', None, 'If training data is precomputed, give a path to pickle object')
flags.DEFINE_string('training_features', None, 'Use precomputed training features from '
                                               './data/features/training_features')

flags.DEFINE_string('test_file', 'dataset.pkl',
                    '.pkl file with saved weights, should be places in ./data')
flags.DEFINE_bool('split_dataset', True, 'Set to true if you do not have a separate dataset')
flags.DEFINE_integer('experiment_num', 1, 'If you run multiple experiments, keep track of experiment number')
flags.DEFINE_bool('save_dataset', False, 'Set to true if you wish to save the computed frequency averages')
flags.DEFINE_string('saved_file_name', 'dataset', 'Name for saving the dataset, used if save_dataset is true')
flags.DEFINE_bool('save_results', False, 'Appends results to results.txt and images in img folder')


def main(_argv):
    print("App started...")

    solver_object = FrequencySolver(num_iter=FLAGS.num_iter, features=FLAGS.features)

    reals_path, fakes_path = (
    FLAGS.data_path + '/real', FLAGS.data_path + '/fakes') if FLAGS.training_features is None else (None, None)

    solver_object(reals_path=reals_path, fakes_path=fakes_path,
                  training_features=FLAGS.training_features)
    print("Initialization finished\n")

    if FLAGS.save_dataset:
        solver_object.save_dataset(file_name=FLAGS.saved_file_name)
        print("Weights saved")

    solver_object.train(test_file=FLAGS.test_file, split_dataset=FLAGS.split_dataset)
    # solver_object.train_NN(testset_path='ff_test_199_crop.pkl')

    # saving
    if solver_object.type == "nn":
        output_name = './data/models/pretrained_NN.pkl'
    else:
        output_name = './data/models/pretrained_SVM_r.pkl'
    output = open(output_name, 'wb')
    pickle.dump(solver_object, output)
    output.close()

    print("Training finished")

    # solver_object.visualize()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
