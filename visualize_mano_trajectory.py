import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import argparse
from pathlib import Path
import cv2
import os
from tqdm import tqdm

from wilor.models import MANO
from wilor.configs import get_config
from wilor.utils.renderer import Renderer

# MANO hand skeleton connections (21 joints)
HAND_SKELETON = [
    [0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],  # Index
    [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
    [0, 13], [13, 14], [14, 15], [15, 16],  # Ring
    [0, 17], [17, 18], [18, 19], [19, 20],  # Pinky
]

LIGHT_PURPLE = (0.25098039, 0.274117647, 0.65882353)

def load_mano_model_and_config(cfg_path):
    """Load only MANO model and config, without the entire WiLoR model.
    
    This is much more efficient than loading the full WiLoR model since we only
    need MANO for reconstructing vertices from parameters.
    """
    # Load config
    model_cfg = get_config(cfg_path, update_cachedir=True)
    
    # Update MANO paths to be relative to current directory
    if 'DATA_DIR' in model_cfg.MANO:
        model_cfg.defrost()
        model_cfg.MANO.DATA_DIR = './mano_data/'
        model_cfg.MANO.MODEL_PATH = './mano_data/'
        model_cfg.MANO.MEAN_PARAMS = './mano_data/mano_mean_params.npz'
        model_cfg.freeze()
    
    # Create MANO model directly from config
    mano_cfg = {k.lower(): v for k, v in dict(model_cfg.MANO).items()}
    mano_model = MANO(**mano_cfg)
    
    return mano_model, model_cfg

def load_mano_trajectory(npz_path):
    """Load MANO parameters from numpy file.
    
    Expected format: [nframes, max_n_hands, ...] with valid_hands_mask
    """
    data = np.load(npz_path, allow_pickle=True)
    
    # Load new format: [nframes, max_n_hands, ...]
    if 'valid_hands_mask' not in data:
        raise ValueError(f"Invalid NPZ file format. Expected format with 'valid_hands_mask'. "
                        f"Found keys: {list(data.keys())}")
    
    global_orient = data['global_orient']  # [n_frames, max_n_hands, 1, 3, 3]
    hand_pose = data['hand_pose']  # [n_frames, max_n_hands, 15, 3, 3]
    betas = data['betas']  # [n_frames, max_n_hands, 10]
    cam_translation = data['camera_translation']  # [n_frames, max_n_hands, 3]
    is_right = data['is_right']  # [n_frames, max_n_hands]
    valid_hands_mask = data['valid_hands_mask']  # [n_frames, max_n_hands]
    n_frames = int(data['n_frames'])
    
    # Convert to per-frame format (list of arrays, one per frame)
    global_orient_per_frame = []
    hand_pose_per_frame = []
    betas_per_frame = []
    cam_translation_per_frame = []
    is_right_per_frame = []
    
    for frame_idx in range(n_frames):
        # Get valid hands for this frame
        valid_mask = valid_hands_mask[frame_idx]  # [max_n_hands]
        n_valid = valid_mask.sum()
        
        if n_valid > 0:
            global_orient_per_frame.append(global_orient[frame_idx, valid_mask])
            hand_pose_per_frame.append(hand_pose[frame_idx, valid_mask])
            betas_per_frame.append(betas[frame_idx, valid_mask])
            cam_translation_per_frame.append(cam_translation[frame_idx, valid_mask])
            is_right_per_frame.append(is_right[frame_idx, valid_mask])
        else:
            # Empty frame
            global_orient_per_frame.append(np.zeros((0, 1, 3, 3)))
            hand_pose_per_frame.append(np.zeros((0, 15, 3, 3)))
            betas_per_frame.append(np.zeros((0, 10)))
            cam_translation_per_frame.append(np.zeros((0, 3)))
            is_right_per_frame.append(np.zeros((0,), dtype=bool))
    
    return {
        'global_orient_per_frame': global_orient_per_frame,
        'hand_pose_per_frame': hand_pose_per_frame,
        'betas_per_frame': betas_per_frame,
        'camera_translation_per_frame': cam_translation_per_frame,
        'is_right_per_frame': is_right_per_frame,
        'n_frames': n_frames,
        'valid_hands_mask': valid_hands_mask
    }

def reconstruct_vertices_from_mano(mano_model, global_orient, hand_pose, betas, device):
    """Reconstruct vertices from MANO parameters."""
    batch_size = global_orient.shape[0]
    
    # Reshape parameters for MANO
    global_orient_tensor = torch.from_numpy(global_orient).float().to(device)  # [B, 1, 3, 3]
    hand_pose_tensor = torch.from_numpy(hand_pose).float().to(device)  # [B, 15, 3, 3]
    betas_tensor = torch.from_numpy(betas).float().to(device)  # [B, 10]
    
    # Reshape for MANO forward pass
    global_orient_tensor = global_orient_tensor.reshape(batch_size, -1, 3, 3)
    hand_pose_tensor = hand_pose_tensor.reshape(batch_size, -1, 3, 3)
    betas_tensor = betas_tensor.reshape(batch_size, -1)
    
    # Forward through MANO
    with torch.no_grad():
        mano_output = mano_model(
            global_orient=global_orient_tensor,
            hand_pose=hand_pose_tensor,
            betas=betas_tensor,
            pose2rot=False
        )
        vertices = mano_output.vertices.cpu().numpy()  # [B, 778, 3]
        joints = mano_output.joints.cpu().numpy()  # [B, 21, 3]
    
    return vertices, joints

def plot_static_trajectory(vertices_list, joints_list, cam_translation_list, output_path=None):
    """Create a static 3D plot showing the entire trajectory from reconstructed meshes."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Stack joints into 3D array: [total_hands, 21, 3]
    if len(joints_list) == 0:
        print("Warning: No joints to plot")
        return
    all_joints = np.stack(joints_list, axis=0)  # [total_hands, 21, 3]
    all_cam_t = np.stack(cam_translation_list, axis=0) if len(cam_translation_list) > 0 else np.array([])  # [total_hands, 3]
    
    n_hands = all_joints.shape[0]
    
    # Plot trajectory of wrist (joint 0)
    wrist_traj = all_joints[:, 0, :]
    ax.plot(wrist_traj[:, 0], wrist_traj[:, 1], wrist_traj[:, 2], 
            'b-', linewidth=2, alpha=0.6, label='Wrist trajectory')
    
    # Plot start and end positions
    ax.scatter(*wrist_traj[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*wrist_traj[-1], color='red', s=100, marker='s', label='End')
    
    # Plot a few key hand poses along the trajectory
    step = max(1, n_hands // 10)  # Show ~10 poses
    for i in range(0, n_hands, step):
        frame_joints = all_joints[i]
        # Plot hand skeleton for this frame
        for connection in HAND_SKELETON:
            start_idx, end_idx = connection
            ax.plot([frame_joints[start_idx, 0], frame_joints[end_idx, 0]],
                   [frame_joints[start_idx, 1], frame_joints[end_idx, 1]],
                   [frame_joints[start_idx, 2], frame_joints[end_idx, 2]],
                   'gray', alpha=0.3, linewidth=1)
        # Plot joints
        ax.scatter(frame_joints[:, 0], frame_joints[:, 1], frame_joints[:, 2],
                  c='blue', s=20, alpha=0.5)
    
    # Plot camera translation trajectory
    if all_cam_t.size > 0 and len(all_cam_t.shape) == 2:
        ax.plot(all_cam_t[:, 0], all_cam_t[:, 1], all_cam_t[:, 2],
                'r--', linewidth=1.5, alpha=0.5, label='Camera translation')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Hand Trajectory Visualization ({n_hands} hands)')
    ax.legend()
    ax.grid(True)
    
    # Set equal aspect ratio
    max_range = np.array([all_joints[:, :, 0].max() - all_joints[:, :, 0].min(),
                          all_joints[:, :, 1].max() - all_joints[:, :, 1].min(),
                          all_joints[:, :, 2].max() - all_joints[:, :, 2].min()]).max() / 2.0
    mid_x = (all_joints[:, :, 0].max() + all_joints[:, :, 0].min()) * 0.5
    mid_y = (all_joints[:, :, 1].max() + all_joints[:, :, 1].min()) * 0.5
    mid_z = (all_joints[:, :, 2].max() + all_joints[:, :, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved static visualization to {output_path}")
    else:
        plt.show()

def animate_trajectory(vertices_list, joints_list, cam_translation_list, output_path=None, interval=50):
    """Create an animated 3D plot showing the hand moving through space."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if len(joints_list) == 0:
        print("Warning: No joints to animate")
        return None
    
    # Stack joints into 3D array: [total_hands, 21, 3]
    all_joints = np.stack(joints_list, axis=0)  # [total_hands, 21, 3]
    all_cam_t = np.stack(cam_translation_list, axis=0) if len(cam_translation_list) > 0 else np.array([])  # [total_hands, 3]
    n_hands = all_joints.shape[0]
    
    # Initialize plot elements
    wrist_traj_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6, label='Wrist trajectory')
    hand_skeleton_lines = []
    hand_joints_scatter = ax.scatter([], [], [], c='blue', s=50, alpha=0.8)
    wrist_current, = ax.plot([], [], [], 'ro', markersize=10, label='Current wrist')
    cam_traj_line, = ax.plot([], [], [], 'r--', linewidth=1.5, alpha=0.5, label='Camera translation')
    
    # Set axis limits based on all data
    max_range = np.array([all_joints[:, :, 0].max() - all_joints[:, :, 0].min(),
                          all_joints[:, :, 1].max() - all_joints[:, :, 1].min(),
                          all_joints[:, :, 2].max() - all_joints[:, :, 2].min()]).max() / 2.0
    mid_x = (all_joints[:, :, 0].max() + all_joints[:, :, 0].min()) * 0.5
    mid_y = (all_joints[:, :, 1].max() + all_joints[:, :, 1].min()) * 0.5
    mid_z = (all_joints[:, :, 2].max() + all_joints[:, :, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hand Trajectory Animation')
    ax.legend()
    ax.grid(True)
    
    def update(frame):
        # Update wrist trajectory (show up to current frame)
        wrist_traj = all_joints[:frame+1, 0, :]
        wrist_traj_line.set_data(wrist_traj[:, 0], wrist_traj[:, 1])
        wrist_traj_line.set_3d_properties(wrist_traj[:, 2])
        
        # Update camera translation trajectory
        if all_cam_t.size > 0 and len(all_cam_t.shape) == 2:
            cam_traj = all_cam_t[:frame+1, :]
            cam_traj_line.set_data(cam_traj[:, 0], cam_traj[:, 1])
            cam_traj_line.set_3d_properties(cam_traj[:, 2])
        
        # Update current hand pose
        frame_joints = all_joints[frame]
        
        # Remove old skeleton lines
        for line in hand_skeleton_lines:
            line.remove()
        hand_skeleton_lines.clear()
        
        # Draw hand skeleton
        for connection in HAND_SKELETON:
            start_idx, end_idx = connection
            line, = ax.plot([frame_joints[start_idx, 0], frame_joints[end_idx, 0]],
                           [frame_joints[start_idx, 1], frame_joints[end_idx, 1]],
                           [frame_joints[start_idx, 2], frame_joints[end_idx, 2]],
                           'blue', linewidth=2, alpha=0.8)
            hand_skeleton_lines.append(line)
        
        # Update joints scatter
        hand_joints_scatter._offsets3d = (frame_joints[:, 0], frame_joints[:, 1], frame_joints[:, 2])
        
        # Update current wrist position
        wrist_current.set_data([frame_joints[0, 0]], [frame_joints[0, 1]])
        wrist_current.set_3d_properties([frame_joints[0, 2]])
        
        ax.set_title(f'Hand Trajectory Animation - Hand {frame+1}/{n_hands}')
        
        return wrist_traj_line, cam_traj_line, hand_joints_scatter, wrist_current, *hand_skeleton_lines
    
    anim = animation.FuncAnimation(fig, update, frames=n_hands, interval=interval, blit=False)
    
    if output_path:
        print(f"Saving animation to {output_path}...")
        anim.save(output_path, writer='ffmpeg', fps=1000//interval, bitrate=1800)
        print(f"Saved animation to {output_path}")
    else:
        plt.show()
    
    return anim

def render_hand_mesh_sequence(hand_data, mano_model, renderer, model_cfg, output_dir, device, 
                              render_res=(640, 480), focal_length=5000):
    """Render mesh sequence for a single hand."""
    os.makedirs(output_dir, exist_ok=True)
    
    vertices_list = hand_data['vertices']
    joints_list = hand_data['joints']
    cam_t_list = hand_data['cam_t']
    frame_nums = hand_data['frame_nums']
    global_orient_list = hand_data['global_orient']
    hand_pose_list = hand_data['hand_pose']
    betas_list = hand_data['betas']
    is_right_list = hand_data['is_right']
    
    n_frames = len(vertices_list)
    
    # Sort frame numbers to ensure correct order
    sorted_indices = sorted(range(n_frames), key=lambda i: frame_nums[i])
    
    for idx in tqdm(sorted_indices, desc=f"Rendering hand meshes"):
        frame_num = frame_nums[idx]
        
        # Get parameters for this frame
        global_orient = global_orient_list[idx]  # [1, 3, 3]
        hand_pose = hand_pose_list[idx]  # [15, 3, 3]
        betas = betas_list[idx]  # [10]
        cam_t = cam_t_list[idx]  # [3]
        is_right_val = is_right_list[idx]
        
        # Reconstruct vertices (need to add batch dimension)
        global_orient_batch = global_orient[np.newaxis, ...]  # [1, 1, 3, 3]
        hand_pose_batch = hand_pose[np.newaxis, ...]  # [1, 15, 3, 3]
        betas_batch = betas[np.newaxis, ...]  # [1, 10]
        
        vertices, joints = reconstruct_vertices_from_mano(
            mano_model, global_orient_batch, hand_pose_batch, betas_batch, device
        )
        
        verts = vertices[0].copy()  # Remove batch dimension
        if not is_right_val:
            verts[:, 0] = -verts[:, 0]
        
        # Render single hand
        misc_args = dict(
            mesh_base_color=LIGHT_PURPLE,
            scene_bg_color=(1, 1, 1),
            focal_length=focal_length,
        )
        
        cam_view = renderer.render_rgba(
            verts,
            cam_t=cam_t.copy(),
            render_res=render_res,
            is_right=is_right_val,
            **misc_args
        )
        
        # Save rendered image
        output_path = os.path.join(output_dir, f'frame_{frame_num:04d}.png')
        cam_view_uint8 = (cam_view[:, :, :3] * 255).astype(np.uint8)
        cam_view_bgr = cv2.cvtColor(cam_view_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, cam_view_bgr)
    
    print(f"Rendered {n_frames} mesh frames to {output_dir}")

def create_video_from_frames(frame_dir, output_video_path, fps=15):
    """Create a video by stitching together rendered mesh frames."""
    # Get all frame files and sort them
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.png') or f.endswith('.jpg') and f.startswith('frame_')])
    
    if len(frame_files) == 0:
        print(f"Warning: No frame files found in {frame_dir}")
        return
    
    # Read first frame to get dimensions
    first_frame_path = os.path.join(frame_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    if first_frame is None:
        print(f"Warning: Could not read first frame from {first_frame_path}")
        return
    
    height, width, _ = first_frame.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return
    
    # Write all frames to video
    for frame_file in tqdm(frame_files, desc="Creating video from frames"):
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Resize frame if dimensions don't match
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
    
    video_writer.release()
    print(f"Created video: {output_video_path} ({len(frame_files)} frames at {fps} fps)")


def main():
    parser = argparse.ArgumentParser(description='Visualize hand trajectory from MANO parameters NPZ file')
    parser.add_argument('--type', type=str, default='gluing', help='Type of video')
    parser.add_argument('--trajectory_file', type=str, 
                       default='../processed_videos/gluing/hand_trajectory.npz',
                       help='Path to hand_trajectory.npz file')
    parser.add_argument('--mode', type=str, 
                       choices=['static', 'animate', 'mesh', 'all'],
                       default='all', help='Visualization mode')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save output images/videos')
    parser.add_argument('--animation_interval', type=int, default=50,
                       help='Animation interval in milliseconds (for 3D trajectory animation)')
    parser.add_argument('--video_fps', type=float, default=15,
                       help='FPS for stitched mesh video animation')
    parser.add_argument('--cfg_path', type=str,
                       default='./pretrained_models/model_config.yaml',
                       help='Path to model config (only needed for MANO model and renderer setup)')
    parser.add_argument('--max_frames', type=int, default=None,
                       help='Maximum number of frames to process')
    parser.add_argument('--render_res', type=int, nargs=2, default=[640, 480],
                       help='Rendering resolution [width, height]')
    parser.add_argument('--focal_length', type=float, default=5000,
                       help='Focal length for rendering')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Input directory to save the video')
    
    args = parser.parse_args()

    args.output_dir = f'/home/ap977/GRILL/new_project/processed_videos/{args.type}/visualizations'
    args.input_dir = f'/home/ap977/GRILL/new_project/processed_videos/{args.type}'
    
    # Load trajectory data
    print(f"Loading MANO trajectory from {args.trajectory_file}...")
    mano_data = load_mano_trajectory(args.trajectory_file)
    print(f"Loaded trajectory data: {mano_data['n_frames']} frames")
    
    # Load MANO model for reconstruction (only MANO, not the entire WiLoR model)
    print("Loading MANO model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mano_model, model_cfg = load_mano_model_and_config(args.cfg_path)
    mano_model = mano_model.to(device)
    mano_model.eval()
    
    # Setup renderer
    renderer = Renderer(model_cfg, faces=mano_model.faces)
    
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Group hands by hand index across frames and reconstruct vertices
    print("Reconstructing vertices from MANO parameters and grouping by hand...")
    vertices_list = []  # For combined visualization
    joints_list = []  # For combined visualization
    cam_translation_list = []  # For combined visualization
    hand_groups = {}  # {hand_idx: {'vertices': [], 'joints': [], 'cam_t': [], 'frame_nums': []}}
    
    global_orient_per_frame = mano_data['global_orient_per_frame']
    hand_pose_per_frame = mano_data['hand_pose_per_frame']
    betas_per_frame = mano_data['betas_per_frame']
    cam_translation_per_frame = mano_data['camera_translation_per_frame']
    n_frames = mano_data['n_frames']
    
    for frame in tqdm(range(n_frames)):
        if args.max_frames and frame >= args.max_frames:
            break
            
        global_orient = global_orient_per_frame[frame]
        hand_pose = hand_pose_per_frame[frame]
        betas = betas_per_frame[frame]
        cam_t = cam_translation_per_frame[frame]
        
        if global_orient.shape[0] == 0:
            continue
        
        vertices, joints = reconstruct_vertices_from_mano(
            mano_model=mano_model,
            global_orient=global_orient,
            hand_pose=hand_pose,
            betas=betas,
            device=device
        )
        
        # Apply is_right transformation
        is_right = mano_data['is_right_per_frame'][frame]
        for i in range(vertices.shape[0]):
            if not is_right[i]:
                vertices[i, :, 0] = -vertices[i, :, 0]
                joints[i, :, 0] = -joints[i, :, 0]
        
        # Group by hand index within frame and store for combined visualization
        for i in range(vertices.shape[0]):
            hand_idx = i
            if hand_idx not in hand_groups:
                hand_groups[hand_idx] = {
                    'vertices': [],
                    'joints': [],
                    'cam_t': [],
                    'frame_nums': [],
                    'global_orient': [],
                    'hand_pose': [],
                    'betas': [],
                    'is_right': []
                }
            
            # Store for hand-specific visualization
            hand_groups[hand_idx]['vertices'].append(vertices[i] + cam_t[i])
            hand_groups[hand_idx]['joints'].append(joints[i] + cam_t[i])
            hand_groups[hand_idx]['cam_t'].append(cam_t[i])
            hand_groups[hand_idx]['frame_nums'].append(frame)
            hand_groups[hand_idx]['global_orient'].append(global_orient[i])
            hand_groups[hand_idx]['hand_pose'].append(hand_pose[i])
            hand_groups[hand_idx]['betas'].append(betas[i])
            hand_groups[hand_idx]['is_right'].append(bool(is_right[i]))
            
            # Also store for combined visualization
            vertices_list.append(vertices[i] + cam_t[i])
            joints_list.append(joints[i] + cam_t[i])
            cam_translation_list.append(cam_t[i])
    
    print(f"Found {len(hand_groups)} unique hands across frames")
    
    # Generate visualizations for each hand
    for hand_idx, hand_data in hand_groups.items():
        hand_dir = None
        if args.output_dir:
            hand_dir = str(Path(args.output_dir) / f'hand_{hand_idx}')
            Path(hand_dir).mkdir(parents=True, exist_ok=True)
            print(f"\nProcessing hand {hand_idx} ({len(hand_data['vertices'])} frames)...")
        
        # Prepare data for this hand
        hand_vertices = hand_data['vertices']
        hand_joints = hand_data['joints']
        hand_cam_t = hand_data['cam_t']
        
        # Generate static trajectory
        if args.mode in ['static', 'all']:
            output_path = None
            if hand_dir:
                output_path = str(Path(hand_dir) / 'trajectory_static.png')
            plot_static_trajectory(hand_vertices, hand_joints, hand_cam_t, output_path)
        
        # Generate mesh renders for this hand
        mesh_output_dir = None
        if args.mode in ['mesh', 'all', 'animate']:
            mesh_output_dir = str(Path(hand_dir) / 'mesh_renders') if hand_dir else f'mesh_renders_hand_{hand_idx}'
            render_hand_mesh_sequence(
                hand_data, mano_model, renderer, model_cfg, mesh_output_dir, device,
                render_res=tuple(args.render_res),
                focal_length=args.focal_length
            )
        
        # Generate animated trajectory by stitching mesh frames
        if args.mode in ['animate', 'all']:
            if mesh_output_dir and os.path.exists(mesh_output_dir):
                output_path = None
                if hand_dir:
                    output_path = str(Path(hand_dir) / 'trajectory_animation.mp4')
                    create_video_from_frames(mesh_output_dir, output_path, fps=args.video_fps)
            else:
                # Fallback to 3D trajectory animation if mesh rendering was skipped
                output_path = None
                if hand_dir:
                    output_path = str(Path(hand_dir) / 'trajectory_animation.mp4')
                animate_trajectory(hand_vertices, hand_joints, hand_cam_t, output_path, args.animation_interval)
    
    # Also generate combined visualization (all hands together)
    if args.output_dir and len(hand_groups) > 1:
        print(f"\nGenerating combined visualization (all hands)...")
        combined_dir = str(Path(args.output_dir) / 'all_hands_combined')
        Path(combined_dir).mkdir(parents=True, exist_ok=True)

        create_video_from_frames(args.input_dir, args.output_dir + "/all_hands_combined.mp4", fps=args.video_fps)
        
        if args.mode in ['static', 'all']:
            output_path = str(Path(combined_dir) / 'trajectory_static.png')
            plot_static_trajectory(vertices_list, joints_list, cam_translation_list, output_path)
        
        if args.mode in ['animate', 'all']:
            # For combined view, use 3D trajectory animation since we have multiple hands
            output_path = str(Path(combined_dir) / 'trajectory_animation.mp4')
            animate_trajectory(vertices_list, joints_list, cam_translation_list, output_path, args.animation_interval)
    
    if args.mode == 'all' and not args.output_dir:
        plt.show()

if __name__ == '__main__':
    main()

