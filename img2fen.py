import argparse
import logging
import chessboard_detection as cd
from nn_utils import ChessCNNv3, load_model
from chessboard_utils import generate_fen

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - [%(levelname)s] - %(funcName)s(%(lineno)d) - %(message)s')

    parser = argparse.ArgumentParser(description='Generate FEN from chessboard image.')
    parser.add_argument('input',
                        type=str,
                        help='The path to the image to generate the FEN from',
    )
    parser.add_argument('--slice',
                        dest='slice',
                        action='store_true',
                        help='Just slice the image into squares instead of detecting the squares'
    )
    parser.add_argument('--show', '-s',
                        dest='show',
                        action='store_false',
                        help='Show the image with the detected squares'
    )
    parser.add_argument('--vertical-error',
                        dest='vertical_error',
                        type=float,
                        default=0.5,
                        help='The vertical error to use for the detection'
    )
    parser.add_argument('--horizontal-error',
                        dest='horizontal_error',
                        type=float,
                        default=0.5,
                        help='The horizontal error to use for the detection'
    )
    parser.add_argument('--threshold-x',
                        dest='threshold_x',
                        type=int,
                        default=20,
                        help='The threshold to use for merging lines in the x direction'
    )
    parser.add_argument('--threshold-y',
                        dest='threshold_y',
                        type=int,
                        default=20,
                        help='The threshold to use for merging lines in the y direction'
    )
    parser.add_argument('--lines-threshold',
                        dest='lines_threshold',
                        type=int,
                        default=400,
                        help='The threshold to use for Hough lines detection'
    )
    parser.add_argument('--frame',
                        action='store_false',
                        help='Whether to add a frame to the image'
    )
    parser.add_argument('--frame-thickness',
                        dest='frame_thickness',
                        type=int,
                        default=5,
                        help='The thickness of the frame to add to the image'
    )
    parser.add_argument('--find-board', '-f',
                        dest='find_board',
                        action='store_false',
                        help='Whether to find the board in the image'
    )
    parser.add_argument('--zoom-size',
                        dest='zoom_size',
                        type=int,
                        default=5,
                        help='The zoom size to use for cropping the image'
    )
    parser.add_argument('--profile', '-p', 
                        type=int,
                        help='The profile to use for the detection'
    )
    parser.add_argument('--model', '-m',
                        type=str,
                        default='model/model85.pth',
                        help='The path to the model to use for the detection'
    )

    args = parser.parse_args()

    # Load the model
    model, model_transform = load_model(args.model)
    logging.info(f'Loaded model from {args.model}.')

    if args.find_board:
        boards = cd.find_chessboards(args.input, args.show)
    else:
        boards = [args.input]

    logging.info(f'Found {len(boards)} boards in the image.')

    generated_fens = dict()

    for i, board in enumerate(boards):
        try:
            if args.slice:
                logging.info('Slicing the image into squares.')
                squares = cd.slice_squares(board, **vars(args))
            else:
                logging.info('Detecting the squares in the image.')
                squares = cd.detect_squares(board, **vars(args))
        except Exception as e:
            logging.error(f'Error: {e}')
            continue
        generated_fens[i] = generate_fen(squares, model, model_transform)
    if not generated_fens:
        logging.error('No FENs generated.')
        exit(1)
    
    for i, fen in enumerate(generated_fens.values()):
        print(f'FEN {i}: {fen}')
