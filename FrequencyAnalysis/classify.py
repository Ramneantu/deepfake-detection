from libs.FrequencySolver import FrequencySolver
from absl import app, flags
from absl.flags import FLAGS

# Choose the number of input images from each dataset, default is set to 1000
flags.DEFINE_integer('num_files', 1000, 'Number of images that will be used form EACH dataset, e.g. if set to 500, '
                                       'there will be 500 fake images and 500 real images used')
flags.DEFINE_integer('features', 300, 'Number of features used for training')
# Indicate path to training/test set
flags.DEFINE_string('data_path', None, 'Path to dataset for computing input features')

# flags.DEFINE_bool('compute_data', True, 'If training data is not saved, compute new training data using given paths')
# flags.DEFINE_string('saved_data', None, 'If training data is precomputed, give a path to pickle object')
flags.DEFINE_string('training_features', None, 'Use precomputed training features from '
                                               './data/features/training_features')

# flags.DEFINE_string('test_file', 'dataset.pkl',
#                     '.pkl file with saved weights, should be places in ./data')
flags.DEFINE_string('test_features', None, 'Use precomputed training features for testing from ./data/features/test_features')

# flags.DEFINE_bool('split_dataset', True, 'Set to true if you do not have a separate dataset')
flags.DEFINE_string('save_features', None, 'Give a name for the computed features file, should end in .pkl')
flags.DEFINE_string('save_test_features', None, 'Give a name for computed test features file, should end in .pkl')

flags.DEFINE_string('solver', 'svm', 'Give the name of the solver. Change to nn if you want to use the FreqNN')

flags.DEFINE_bool('save_results', False, 'Appends results to results.txt and images in img folder')


def main(_argv):
    print("App started...")

    solver_object = FrequencySolver(features=FLAGS.features)

    reals_path, fakes_path = (
    FLAGS.data_path + '/train/real', FLAGS.data_path + '/train/fake') if FLAGS.training_features is None else (None, None)

    solver_object(reals_path=reals_path, fakes_path=fakes_path,
                  training_features=FLAGS.training_features)
    print("Initialization finished\n")

    if FLAGS.save_features is not None:
        solver_object.save_dataset(file_name=FLAGS.save_features)
        print("Features saved")

    if FLAGS.solver == "svm":
        solver_object.train()
    elif FLAGS.solver == "nn":
        solver_object.train_NN()

    solver_object.test(test_features=FLAGS.test_features)

    if FLAGS.save_test_features is not None:
        solver_object.save_dataset(file_name=FLAGS.save_test_features, type="test")

    print("App finished\n")

    # saving
    # if solver_object.type == "nn":
    #     output_name = './data/models/frequency_NN_obj.pkl'
    # else:
    #     output_name = './data/models/frequency_SVM_r_obj.pkl'
    # output = open(output_name, 'wb')
    # pickle.dump(solver_object, output)
    # output.close()

    # solver_object.visualize()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
