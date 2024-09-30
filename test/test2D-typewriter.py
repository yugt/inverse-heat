import argparse
import pandas as pd
import resource, os, sys

# Add src/ to the module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from moviepy.editor import ImageSequenceClip, concatenate_videoclips
from inverse_heat2D import inverse_solver
from frame_drawer import Frame_Drawer
from util import prepare_data

# Set memory limit for the program
resource.setrlimit(resource.RLIMIT_AS, (2**34, 2**34))

def save_csv(experiment, df):
    """Save the data to CSV files for logging."""
    prefix = 'log/data'
    pd.DataFrame(df['tbasis'])\
        .to_csv(f'{prefix}/{experiment:03d}-tbasis.csv', index=False, header=False)
    pd.DataFrame(df['grad'])\
        .to_csv(f'{prefix}/{experiment:03d}-t_grad.csv', index=False, header=False)
    # t for typewriter (pixel basis)
    pd.DataFrame({
        'loss': df['loss'],
        'error': df['error']
    }).to_csv(f'{prefix}/{experiment:03d}-loss_error.csv', index=False)


def png2movie(index, frame_drawer):
    forward = ImageSequenceClip([f'tmp/{index:03d}/forward-{i:03d}.png'
                    for i in range(frame_drawer.t.shape[0])], fps=20)
    inverse = ImageSequenceClip([f'tmp/{index:03d}/epoch-{i:03d}.png'
                    for i in range(frame_drawer.epoch.shape[0])], fps=30)
    ## Write the clip to a video file
    combined = concatenate_videoclips([forward, inverse])
    combined.write_videofile(f'log/video/train_log{index:03d}.mp4',
                                codec='libx264')

def main(args):
    # Prepare the training dataset with grid size `J`
    trainset = prepare_data(J=32)

    # Setup experiment ID from the provided argument
    experiment = args.experiment
    sys.stderr.write(f"Running experiment {experiment}\n")

    # Initialize solver and data variables
    solver, df = None, None
    frame_drawer = Frame_Drawer(None, None, None, epoch=args.epochs)

    try:
        # Create temporary directory for saving intermediate results
        os.system(f'mkdir -p tmp/{experiment:03d}')

        # Run the inverse solver with the specified sensor type
        solver, df = inverse_solver(
            (trainset[experiment][0].squeeze().flip(dims=(0,)) / 100 + 0.01),
            epoch=args.epochs,
            path=f'tmp/{experiment:03d}',
            frame_drawer=frame_drawer,
            sensor_type=args.sensor_type
        )

        # Save results to CSV files
        save_csv(experiment, df)

        # Generate video from PNG frames
        png2movie(experiment, frame_drawer)

        # Move final epoch image to log directory
        os.system(f'mv tmp/{experiment:03d}/epoch-{len(frame_drawer.epoch)-1}.png\
                log/image/{experiment:03d}-epoch-{len(frame_drawer.epoch)-1}.png')

        # Cleanup temporary files
        os.system(f'rm -r tmp/{experiment:03d}')

    except Exception as e:
        print(f'Experiment {experiment:03d} failed with error: {e}')

    finally:
        # Clean up solver and data variables
        del solver, df

if __name__ == "__main__":
    # Argument parser for configuration
    parser = argparse.ArgumentParser(description='Run inverse heat equation experiments.')
    parser.add_argument('--experiment', type=int, default=0, help='Experiment index in the training set')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--sensor-type', type=str, choices=['moving', 'static'], default='moving',
                        help='Type of sensors to use: "moving" (4 sensors) or "static" (16 sensors)')

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)