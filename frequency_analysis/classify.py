
from libs.FrequencySolver import FrequencySolver
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_integer('num_iter', 500, 'Number of images that will be used form EACH dataset, e.g. if set to 500, '
                                      'there will be 500 fake images and 500 real images used')
flags.DEFINE_integer('features', 300, 'Number of features used for training')
flags.DEFINE_string('reals_path', None, 'Path to dataset containing real images')
flags.DEFINE_string('fakes_path', None, 'Path to dataset containing fake images')
flags.DEFINE_bool('compute_data', True, 'If training data is not saved, compute new training data using given paths')
flags.DEFINE_string('saved_data', None, 'If training data is precomputed, give a path to pickle object')
flags.DEFINE_bool('crop', False, 'Set to true if you want to crop the face area from the images')
flags.DEFINE_string('test_file', 'dataset.pkl',
                    '.pkl file with saved weights, should be places in ./data')
flags.DEFINE_bool('split_dataset', True, 'Set to true if you do not have a separate dataset')
flags.DEFINE_integer('experiment_num', 1, 'If you run multiple experiments, keep track of experiment number')
flags.DEFINE_bool('save_dataset', False, 'Set to true if you wish to save the computed frequency averages')
flags.DEFINE_string('saved_file_name', 'dataset', 'Name for saving the dataset, used if save_dataset is true')
flags.DEFINE_bool('save_results', False, 'Appends results to results.txt and images in img folder')


def main(_argv):
    logging.info("App started...experiment {}".format(FLAGS.experiment_num))
    logging.info("Started preocessing images...")
    solver_object = FrequencySolver(num_iter=FLAGS.num_iter, features=FLAGS.features)

    solver_object(compute_data=FLAGS.compute_data, reals_path=FLAGS.reals_path, fakes_path=FLAGS.fakes_path,
                  saved_data=FLAGS.saved_data, crop=FLAGS.crop)
    logging.info("Initialization finished")

    # solver_object.train(test_file=FLAGS.test_file, split_dataset=FLAGS.split_dataset)
    solver_object.train_NN()
    logging.info("Training finished")

    # solver_object.visualize()

    if FLAGS.save_dataset:
        solver_object.save_dataset(file_name=FLAGS.saved_file_name)
        logging.info("Weights saved")


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
