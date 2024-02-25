import datashader as ds
import datashader.transfer_functions as tf
from colorcet import fire
import pandas as pd
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from jaxtyping import Float
from epsilon_transformers.analysis.activation_analysis import find_msp_subspace_in_residual_stream

from epsilon_transformers.process.processes import ZeroOneR
from epsilon_transformers.training.configs import RawModelConfig

def _project_to_simplex(points: Float[np.ndarray, "num_points num_states"]):
    """Project points onto the 2-simplex (equilateral triangle in 2D)."""
    x = points[:, 1] + 0.5 * points[:, 2]
    y = (np.sqrt(3) / 2) * points[:, 2]
    return x, y

# Combine aggregated channels into RGB images
def _combine_channels_to_rgb(agg_r, agg_g, agg_b):
    img_r = tf.shade(agg_r, cmap=['black', 'red'], how='linear')
    img_g = tf.shade(agg_g, cmap=['black', 'green'], how='linear')
    img_b = tf.shade(agg_b, cmap=['black', 'blue'], how='linear')

    img_r = tf.spread(img_r, px=1, shape='circle')
    img_g = tf.spread(img_g, px=1, shape='circle')
    img_b = tf.spread(img_b, px=1, shape='circle')

    # Combine using numpy
    r_array = np.array(img_r.to_pil()).astype(np.float64)
    g_array = np.array(img_g.to_pil()).astype(np.float64)
    b_array = np.array(img_b.to_pil()).astype(np.float64)
    
    # Stack arrays into an RGB image (ignoring alpha channel for simplicity)
    rgb_image = np.stack([r_array[:,:,0], g_array[:,:,1], b_array[:,:,2]], axis=-1)
    
    return Image.fromarray(np.uint8(rgb_image))

# TODO: I changed up the code for this to something which makes sense to me (creating panda dataframes from ground truth and predicted simplex. Check to see if this is what should actually be done)
def generate_belief_state_figures_datashader(belief_states, all_beliefs, predicted_beliefs, plot_triangles=False):
    # Projection and DataFrame preparation
    bs_x, bs_y = _project_to_simplex(np.array(belief_states))
    df_gt = pd.DataFrame({'x': bs_x, 'y': bs_y, 'r': belief_states[:, 0], 'g': belief_states[:, 1], 'b': belief_states[:, 2]})

    pb_x, pb_y = _project_to_simplex(np.array(predicted_beliefs))
    df_pb = pd.DataFrame({'x': pb_x, 'y': pb_y, 'r': all_beliefs[:, 0], 'g': all_beliefs[:, 1], 'b': all_beliefs[:, 2]})

    # Create canvas
    cvs = ds.Canvas(plot_width=1000, plot_height=1000, x_range=(-0.1, 1.1), y_range=(-0.1, np.sqrt(3)/2 + 0.1))
    # Aggregate each RGB channel separately for ground truth and predicted beliefs
    agg_funcs = {'r': ds.mean('r'), 'g': ds.mean('g'), 'b': ds.mean('b')}
    agg_gt = {color: cvs.points(df_gt, 'x', 'y', agg_funcs[color]) for color in ['r', 'g', 'b']}
    agg_pb = {color: cvs.points(df_pb, 'x', 'y', agg_funcs[color]) for color in ['r', 'g', 'b']}

    img_gt = _combine_channels_to_rgb(agg_gt['r'], agg_gt['g'], agg_gt['b'])
    img_pb = _combine_channels_to_rgb(agg_pb['r'], agg_pb['g'], agg_pb['b'])

    # Visualization with Matplotlib
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True, facecolor='black')  # Changed 'white' to 'black'
    for ax in axs:
        ax.tick_params(axis='x', colors='black')  # Changed 'black' to 'white'
        ax.tick_params(axis='y', colors='black')  # Changed 'black' to 'white'
        ax.xaxis.label.set_color('black')  # Changed 'black' to 'white'
        ax.yaxis.label.set_color('black')  # Changed 'black' to 'white'
        ax.title.set_color('black')  # Changed 'black' to 'white'
    axs[0].imshow(img_gt)
    axs[1].imshow(img_pb)
    
    axs[0].axis('off')
    axs[1].axis('off')
    title_y_position = -0.1  # Adjust this value to move the title up or down relative to the axes
    fig.text(0.5, title_y_position, 'Ground Truth', ha='center', va='top', transform=axs[0].transAxes, color='white', fontsize=15)  # Changed 'black' to 'white'
    fig.text(0.5, title_y_position, 'Residual Stream', ha='center', va='top', transform=axs[1].transAxes, color='white', fontsize=15)  # Changed 'black' to 'white'

    if plot_triangles:
        for ax in axs:
            ax.plot([0, 0.5, 1, 0], [0, np.sqrt(3)/2, 0, 0], 'white', lw=2)  # Changed 'black' to 'white'

    return fig

if __name__ == "__main__":
    model_config = RawModelConfig(
        d_vocab=2,
        d_model=100,
        n_ctx=10,
        d_head=48,
        n_head=12,
        d_mlp=12,
        n_layers=2,
    )
    model = model_config.to_hooked_transformer(seed=13, device='cpu')
    process = ZeroOneR()
    msp = process.derive_mixed_state_presentation(model.cfg.n_ctx + 1)
    msp.belief_states

    belief_states_reshaped, predicted_beliefs = find_msp_subspace_in_residual_stream(model=model, process=process, num_sequences=5)

    generate_belief_state_figures_datashader(belief_states=msp.belief_states, all_beliefs=belief_states_reshaped, predicted_beliefs=predicted_beliefs, plot_triangles=True)