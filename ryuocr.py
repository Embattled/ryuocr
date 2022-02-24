#! python3
import py


def parse_args(mMain=True, add_help=True):
    import argparse

    # def str2bool(v):
    #     return v.lower() in ("true", "t", "1")
    description = "OCR research program write by Y. Long."
    if mMain:
        parser = argparse.ArgumentParser(
            add_help=add_help, description=description)

        # global, mode select
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--train", action="store_true", help="Train mode.")
        mode.add_argument("--test", action="store_true", help="Test mode.")
        mode.add_argument("--oneoff", action="store_true",
                          help="Train a model and test it without save it to disk.")
        mode.add_argument("--testsscd", action="store_true",
                          help="Show example of sscd.")

        # config path
        parser.add_argument("-c", "--config", type=str,
                            default="config/default.yml", help="Path to config file")

        # for debug and experiment
        parser.add_argument("-l", "--loop", type=int,
                            default=1, help="Run multiple times.")

        # params for oneoff executation
        par_oneoff = parser.add_argument_group(
            "oneoff", "Parameter for oneoff mode")
        par_oneoff.add_argument("--valid_on_testset", action="store_true",
                                help="Using dataset in test group as validation dataset in training")

        # params for test
        par_test = parser.add_argument_group("test", "Parameter for test mode")
        par_test.add_argument("--model", type=str, default="",
                              help="Model used to execute evaluate, if none, using model in config file.")
        par_test.add_argument("--best", action="store_true", help="Whether using best valid accuracy model,'model_best.pt'")

        # params for test sscd
        par_testsscd = parser.add_argument_group(
            "sscd", "Parameter for test sscd.")

        par_testsscd.add_argument("--size", type=int, default=128,
                                  help="resolution of one sscd example.")
        par_testsscd.add_argument("--sscdmargin", type=int, default=0,
                                  help="margin between sscd examples.")
        par_testsscd.add_argument(
            "--sscdshuffle", action="store_true", help="shuffle test sscd")
        par_testsscd.add_argument("--withlabel", action="store_true",
                                  default="whether sscd examples with their labels")
        par_testsscd.add_argument("--sscdcol", type=int, default=8,
                                  help="columns of sscd examples")
        par_testsscd.add_argument("--sscdrow", type=int, default=4,
                                  help="rows of sscd examples")
        par_testsscd.add_argument(
            "--sscdpath", type=str, default="sscd_example.bmp", help="path of sscd examples")
        """
        # params for prediction engine
        parser.add_argument("--use_gpu", type=str2bool, default=True)
        parser.add_argument("--ir_optim", type=str2bool, default=True)
        parser.add_argument("--use_tensorrt", type=str2bool, default=False)
        parser.add_argument("--gpu_mem", type=int, default=8000)

        # params for text detector
        parser.add_argument("--image_dir", type=str)
        parser.add_argument("--det_algorithm", type=str, default='DB')
        parser.add_argument("--det_model_dir", type=str, default=None)
        parser.add_argument("--det_limit_side_len", type=float, default=960)
        parser.add_argument("--det_limit_type", type=str, default='max')

        # DB parmas
        parser.add_argument("--det_db_thresh", type=float, default=0.3)
        parser.add_argument("--det_db_box_thresh", type=float, default=0.5)
        parser.add_argument("--det_db_unclip_ratio", type=float, default=1.6)
        parser.add_argument("--use_dilation", type=bool, default=False)

        # EAST parmas
        parser.add_argument("--det_east_score_thresh", type=float, default=0.8)
        parser.add_argument("--det_east_cover_thresh", type=float, default=0.1)
        parser.add_argument("--det_east_nms_thresh", type=float, default=0.2)

        # params for text recognizer
        parser.add_argument("--rec_algorithm", type=str, default='CRNN')
        parser.add_argument("--rec_model_dir", type=str, default=None)
        parser.add_argument("--rec_image_shape", type=str, default="3, 32, 320")
        parser.add_argument("--rec_char_type", type=str, default='ch')
        parser.add_argument("--rec_batch_num", type=int, default=30)
        parser.add_argument("--max_text_length", type=int, default=25)
        parser.add_argument("--rec_char_dict_path", type=str, default=None)
        parser.add_argument("--use_space_char", type=bool, default=True)
        parser.add_argument("--drop_score", type=float, default=0.5)

        # params for text classifier
        parser.add_argument("--cls_model_dir", type=str, default=None)
        parser.add_argument("--cls_image_shape", type=str, default="3, 48, 192")
        parser.add_argument("--label_list", type=list, default=['0', '180'])
        parser.add_argument("--cls_batch_num", type=int, default=30)
        parser.add_argument("--cls_thresh", type=float, default=0.9)

        parser.add_argument("--enable_mkldnn", type=bool, default=False)
        parser.add_argument("--use_zero_copy_run", type=bool, default=False)
        parser.add_argument("--use_pdserving", type=str2bool, default=False)

        parser.add_argument("--lang", type=str, default='ch')
        parser.add_argument("--det", type=str2bool, default=True)
        parser.add_argument("--rec", type=str2bool, default=True)
        parser.add_argument("--use_angle_cls", type=str2bool, default=False)

        """
        return parser.parse_args()


def main():
    import os.path

    args = parse_args()
    config_path = os.path.abspath(args.config)

    if args.oneoff:
        from py import oneoff
        oneoff.run(config_path, args.loop, args.valid_on_testset)
    elif args.train:
        from py import ryutrain
        ryutrain.run(config_path)
    elif args.test:
        from py import ryutest
        ryutest.run(config_path,args.model,best=args.best)
    elif args.testsscd:
        from py import testsscd
        testsscd.run(config_path, **vars(args))
    else:
        args.print_usage()


if __name__ == "__main__":
    main()
