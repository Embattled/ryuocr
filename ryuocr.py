#! python3
import py


def parse_args(mMain=True, add_help=True):
    import argparse

    # def str2bool(v):
    #     return v.lower() in ("true", "t", "1")

    if mMain:
        parser = argparse.ArgumentParser(add_help=add_help)

        # global 

        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument("--train",action="store_true",help="Train mode.")
        mode.add_argument("--test",action="store_true",help="Test mode.")
        mode.add_argument("--oneoff",action="store_true",help="Train a model and test it without save it to disk.")
        
        parser.add_argument("-c","--config", type=str, default="config/default.yml",help="Path to config file")

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
    args=parse_args()
    print(args)

if __name__ == "__main__":
    main()