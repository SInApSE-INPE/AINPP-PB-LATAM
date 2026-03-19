import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os


def create_animation(target_seq, pred_seq, output_path, fps=5, cmap="viridis"):
    """
    Creates an animation of the forecast evolution.

    Args:
        target_seq (np.array): Sequence of targets (T, H, W).
        pred_seq (np.array): Sequence of predictions (T, H, W).
        output_path (str): Path to save the animation (.gif or .mp4).
        fps (int): Frames per second.
        cmap (str): Colormap.
    """
    # Normalize sequences to same scale for visualization
    vmin = min(np.min(target_seq), np.min(pred_seq))
    vmax = max(np.max(target_seq), np.max(pred_seq))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Initial frames
    im0 = axes[0].imshow(target_seq[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title("Target")

    im1 = axes[1].imshow(pred_seq[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title("Prediction")

    plt.colorbar(im0, ax=axes, fraction=0.025, pad=0.04)

    def update(frame):
        im0.set_data(target_seq[frame])
        im1.set_data(pred_seq[frame])
        fig.suptitle(f"Time Step: {frame}")
        return [im0, im1]

    ani = animation.FuncAnimation(fig, update, frames=len(target_seq), blit=True)

    ani.save(output_path, fps=fps, writer="pillow")  # Use pillow for GIF
    plt.close()
    print(f"Animation saved to {output_path}")
