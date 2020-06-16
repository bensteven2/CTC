class Config(object):
    pass


def get_config(FLAGS, is_train):
    config = Config()

    config.logs_dir = FLAGS.logs_dir
    config.mode = FLAGS.mode
    #  config.out_dir  = FLAGS.out_dir

    config.ClusterNo = 4
    config.batch_size = 1  # The current implementation only supports a batch size equal to unity!

    if is_train:
        config.im_size = 512  # The width and height should be equal. Upper bound of the input image size is limited to the GPU memory.
        config.lr = 1e-4
        config.iteration = 10  # 10e6

        config.ReportInterval = 50
        config.SavingInterval = 500

        config.image_options = {'resize': True, 'resize_size': config.im_size, 'flip': True, 'rotate_stepwise': True,
                                "environment factor": True}
        config.fileformat = 'png'

    else:
        config.batch_size = 1
        config.im_size = 512  # The width and height should be equal. Upper bound of the input image size is limited to the GPU memory.

        config.ReportInterval = None
        config.SavingInterval = None
        config.image_options = {'resize': False, 'resize_size': config.im_size, 'flip': False, 'rotate_stepwise': False,
                                "environment factor": False}
        config.fileformat = 'png'

    return config
