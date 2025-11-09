from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import json
from typing import Dict, Optional
from tqdm import tqdm
from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from wilor.utils.renderer import Renderer, cam_crop_to_full
from ultralytics import YOLO 
LIGHT_PURPLE=(0.25098039,  0.274117647,  0.65882353)

def main():
    parser = argparse.ArgumentParser(description='WiLoR demo code')
    parser.add_argument('--type', type=str, default='gluing', help='Type of video')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False, help='If set, save meshes to disk also')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')

    args = parser.parse_args()
    args.img_folder = f'/home/ap977/GRILL/new_project/videos/{args.type}'
    args.out_folder = f'/home/ap977/GRILL/new_project/processed_videos/{args.type}'

    # Download and load checkpoints
    model, model_cfg = load_wilor(checkpoint_path = './pretrained_models/wilor_final.ckpt' , cfg_path= './pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    renderer_side = Renderer(model_cfg, faces=model.mano.faces)
    
    device   = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model    = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images ends with .jpg or .png
    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]
    img_paths = sorted(img_paths)  # Sort to ensure consistent ordering


    img_paths = img_paths[:50]
    
    # Collect MANO parameters and related data per timestep (frame)
    # Each element in these lists corresponds to one frame and contains arrays for all hands in that frame
    global_orient_per_frame = []
    hand_pose_per_frame = []
    betas_per_frame = []
    cam_t_per_frame = []
    is_right_per_frame = []
    
    # Iterate over all images in folder
    for frame_idx, img_path in enumerate(tqdm(img_paths)):
        img_cv2 = cv2.imread(str(img_path))
        detections = detector(img_cv2, conf = 0, verbose=False)[0]

        detections_list = []
        for det in detections:
            bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            conf_score = det.boxes.conf.data.cpu().detach().squeeze().item()
            is_right = det.boxes.cls.cpu().detach().squeeze().item()
            detections_list.append({
                'bbox': bbox,
                'conf_score': conf_score,
                'is_right': is_right
            })

        left_hand_dets = [det for det in detections_list if det['is_right'] == 0]
        right_hand_dets = [det for det in detections_list if det['is_right'] == 1]

        selected_detections = []
        if len(left_hand_dets) > 0:
            best_left = max(left_hand_dets, key=lambda x: x['conf_score'])
            selected_detections.append(best_left)
        if len(right_hand_dets) > 0:
            best_right = max(right_hand_dets, key=lambda x: x['conf_score'])
            selected_detections.append(best_right)

        bboxes    = []
        is_right  = []
        for det in selected_detections: 
            is_right.append(det['is_right'])
            bboxes.append(det['bbox'])
        
        if len(bboxes) == 0:
            # Empty frame - create empty arrays to maintain frame correspondence
            global_orient_per_frame.append(np.zeros((0, 1, 3, 3)))
            hand_pose_per_frame.append(np.zeros((0, 15, 3, 3)))
            betas_per_frame.append(np.zeros((0, 10)))
            cam_t_per_frame.append(np.zeros((0, 3)))
            is_right_per_frame.append(np.zeros((0,), dtype=bool))
            continue
            
        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        all_verts = []
        all_cam_t = []
        all_right = []
        all_joints= []
        all_kpts  = []
        
        # Collect MANO parameters for all hands in this frame
        frame_global_orient = []
        frame_hand_pose = []
        frame_betas = []
        frame_cam_t = []
        frame_is_right = []
        
        for batch in dataloader: 
            batch = recursive_to(batch, device)
    
            with torch.no_grad():
                out = model(batch) 
                
            multiplier    = (2*batch['right']-1)
            pred_cam      = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center    = batch["box_center"].float()
            box_size      = batch["box_size"].float()
            img_size      = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full     = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Extract MANO parameters
            pred_mano_params = out['pred_mano_params']
            
            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                
                verts  = out['pred_vertices'][n].detach().cpu().numpy()
                joints = out['pred_keypoints_3d'][n].detach().cpu().numpy()
                
                is_right_val = batch['right'][n].cpu().numpy()
                verts[:,0]  = (2*is_right_val-1)*verts[:,0]
                joints[:,0] = (2*is_right_val-1)*joints[:,0]
                cam_t = pred_cam_t_full[n]
                kpts_2d = project_full_img(verts, cam_t, scaled_focal_length, img_size[n])
                
                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right_val)
                all_joints.append(joints)
                all_kpts.append(kpts_2d)
                
                # Extract and save MANO parameters for this hand
                global_orient = pred_mano_params['global_orient'][n].detach().cpu().numpy()  # shape: [1, 3, 3]
                hand_pose = pred_mano_params['hand_pose'][n].detach().cpu().numpy()  # shape: [15, 3, 3]
                betas = pred_mano_params['betas'][n].detach().cpu().numpy()  # shape: [10]
                
                frame_global_orient.append(global_orient)
                frame_hand_pose.append(hand_pose)
                frame_betas.append(betas)
                frame_cam_t.append(cam_t)
                frame_is_right.append(is_right_val)
                
                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_PURPLE, is_right=is_right_val)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{n}.obj'))

        # Save MANO parameters for this frame
        if len(frame_global_orient) > 0:
            # Stack all hands in this frame
            frame_global_orient_array = np.stack(frame_global_orient, axis=0)  # shape: [n_hands, 1, 3, 3]
            frame_hand_pose_array = np.stack(frame_hand_pose, axis=0)  # shape: [n_hands, 15, 3, 3]
            frame_betas_array = np.stack(frame_betas, axis=0)  # shape: [n_hands, 10]
            frame_cam_t_array = np.stack(frame_cam_t, axis=0)  # shape: [n_hands, 3]
            frame_is_right_array = np.stack(frame_is_right, axis=0)  # shape: [n_hands]
        else:
            # Empty frame - create empty arrays with correct shape
            frame_global_orient_array = np.zeros((0, 1, 3, 3))
            frame_hand_pose_array = np.zeros((0, 15, 3, 3))
            frame_betas_array = np.zeros((0, 10))
            frame_cam_t_array = np.zeros((0, 3))
            frame_is_right_array = np.zeros((0,), dtype=bool)
        
        global_orient_per_frame.append(frame_global_orient_array)
        hand_pose_per_frame.append(frame_hand_pose_array)
        betas_per_frame.append(frame_betas_array)
        cam_t_per_frame.append(frame_cam_t_array)
        is_right_per_frame.append(frame_is_right_array)
        
        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_PURPLE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)

            # Overlay image
            input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}.jpg'), 255*input_img_overlay[:, :, ::-1])

    # Save MANO parameters to numpy file, organized as [nframes, nhands, ...]
    if len(global_orient_per_frame) > 0:
        n_frames = len(global_orient_per_frame)
        
        # Find maximum number of hands across all frames
        max_n_hands = max(g.shape[0] for g in global_orient_per_frame) if len(global_orient_per_frame) > 0 else 0
        
        if max_n_hands == 0:
            print("Warning: No hands detected in any frame")
            return
        
        # Initialize arrays with shape [n_frames, max_n_hands, ...]
        global_orient_array = np.zeros((n_frames, max_n_hands, 1, 3, 3))
        hand_pose_array = np.zeros((n_frames, max_n_hands, 15, 3, 3))
        betas_array = np.zeros((n_frames, max_n_hands, 10))
        cam_t_array = np.zeros((n_frames, max_n_hands, 3))
        is_right_array = np.zeros((n_frames, max_n_hands), dtype=bool)
        valid_hands_mask = np.zeros((n_frames, max_n_hands), dtype=bool)  # Mask to indicate valid hands
        
        # Fill arrays frame by frame
        for frame_idx, (go, hp, b, ct, ir) in enumerate(zip(
            global_orient_per_frame, hand_pose_per_frame, betas_per_frame,
            cam_t_per_frame, is_right_per_frame
        )):
            n_hands_in_frame = go.shape[0]
            if n_hands_in_frame > 0:
                # Copy data for valid hands
                global_orient_array[frame_idx, :n_hands_in_frame] = go
                hand_pose_array[frame_idx, :n_hands_in_frame] = hp
                betas_array[frame_idx, :n_hands_in_frame] = b
                cam_t_array[frame_idx, :n_hands_in_frame] = ct
                is_right_array[frame_idx, :n_hands_in_frame] = ir
                valid_hands_mask[frame_idx, :n_hands_in_frame] = True
        
        # Save to numpy file
        output_npy_path = os.path.join(args.out_folder, 'hand_trajectory.npz')
        
        np.savez(output_npy_path, 
                 # Main arrays with shape [n_frames, max_n_hands, ...]
                 global_orient=global_orient_array,      # shape: [n_frames, max_n_hands, 1, 3, 3]
                 hand_pose=hand_pose_array,              # shape: [n_frames, max_n_hands, 15, 3, 3]
                 betas=betas_array,                      # shape: [n_frames, max_n_hands, 10]
                 camera_translation=cam_t_array,         # shape: [n_frames, max_n_hands, 3]
                 is_right=is_right_array,                # shape: [n_frames, max_n_hands]
                 valid_hands_mask=valid_hands_mask,      # shape: [n_frames, max_n_hands] - indicates valid hands
                 n_frames=n_frames,                      # number of frames
                 max_n_hands=max_n_hands)                # maximum number of hands in any frame
        
        # Count total valid hands for reporting
        total_valid_hands = valid_hands_mask.sum()
        
        print(f"Saved MANO parameters to {output_npy_path}")
        print(f"  Total frames: {n_frames}")
        print(f"  Max hands per frame: {max_n_hands}")
        print(f"  Total valid hands: {total_valid_hands}")
        print(f"  Global orient shape: {global_orient_array.shape}")
        print(f"  Hand pose shape: {hand_pose_array.shape}")
        print(f"  Betas shape: {betas_array.shape}")
        print(f"  Camera translation shape: {cam_t_array.shape}")
        print(f"  Valid hands mask shape: {valid_hands_mask.shape}")
        print(f"  Hands per frame: {[g.shape[0] for g in global_orient_per_frame]}")

def project_full_img(points, cam_trans, focal_length, img_res): 
    camera_center = [img_res[0] / 2., img_res[1] / 2.]
    K = torch.eye(3) 
    K[0,0] = focal_length
    K[1,1] = focal_length
    K[0,2] = camera_center[0]
    K[1,2] = camera_center[1]
    points = points + cam_trans
    points = points / points[..., -1:] 
    
    V_2d = (K @ points.T).T 
    return V_2d[..., :-1]

if __name__ == '__main__':
    main()
