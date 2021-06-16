import ryuutils

def parse_args():
    import argparse
    description = "OCR research program write by Y. Long."
    parser = argparse.ArgumentParser(description=description)

    # parser.add_argument("mode",type=str,help="train/test")
    parser.add_argument("-c","--config",type=str, help="Path of config file.")

    return parser.parse_args()


def main():
    # for cmd
    args = parse_args()

    print(args)

if __name__ == '__main__':
    main()