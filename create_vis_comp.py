
import os
import csv
import cv2
import json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import peak_signal_noise_ratio
from tqdm import tqdm

'''
PICKED_SCENES = {
    "re10k": [
        # ('040a26b288e7bda4', 1),
        # ('02ee66b3efbf3b0a', 1),
        # ('01a2277ee817b310', 1),
        # ('002ae53df0e0afe2', 1),
        # ('02ee66b3efbf3b0a', 1),
        # ('01a5cc3805e94c21', 2),
        # above: dep index
        ('0131c9aed0fb3940', 1),
        ('040a26b288e7bda4', 1),
        ('04e2be0415136fa9', 0),
        ('03a78406de1d0993', 1),
        ('0068e97c1c1f61aa', 0),
        ('040895d45bf4e580', 0),



    ],
    "objaverse": [
        # radial
        ('015c200ce786438c8e35ddf635d1e236', 4),
        ('03c0260373c7406ea408c2dec9f8d502', 4),
        ('047ca62b79d140d4b64044db311561d9', 8),
        # spherical
        ('035d9ce9964b42f6bc20514853934d1b', 0),
        ('0050f76a07fa43b7a38e6cef40beb69d', 2),
        ('03e4ff99ecc24e5f8ea49d4e8df876e8', 2),
        # ('006373e3885b472cb5538fc570235fcf', 8),
        
    ],
    "co3d_seen": [
        # 2 views for figure
        ('212_22443_46420', 0),
        # ('12_104_640', 1),
        ('191_20615_38267', 2),
        ('184_19904_39401', 1),
        ('256_27647_53803', 0),
        ('223_23574_49752', 0),
        ('201_21573_40063', 0),

        # 4 views for video
        # ('256_27676_55062', 2),
        # ('118_13853_28129', 0),
        # ('112_13308_25702', 0),
        # ('187_20181_35751', 2),
        # ('164_17988_33493', 3),
        # ('250_26773_54489', 1),
        # ('222_23409_48576', 3),
    ],
    "mixed": {
        "co3d_seen": [
            
            ('256_27676_55062', 1),
            ('118_13853_28129', 0),
        ],
        "re10k": [
            # ('04e2be0415136fa9', 1),
            # ('03a78406de1d0993', 2),
            # above: dep index
            ('002ae53df0e0afe2', 1),
            ('02ee66b3efbf3b0a', 1),
            



        ],
        "objaverse": [
            ('02ca74ef6a1b4ec386c11048603f0e98', 4),
            ('002aec05c41342dea61828f67d340d2d', 8)
        ],
    }
}

'''
PICKED_SCENES = {
    "re10k": [
        ('04e2be0415136fa9', 1),
        ('03a78406de1d0993', 2),
        ('040a26b288e7bda4', 1),
        ('02ee66b3efbf3b0a', 1),
        ('01a2277ee817b310', 1),
        ('002ae53df0e0afe2', 1),
        ('02ee66b3efbf3b0a', 1),
        ('01a5cc3805e94c21', 2),
    ],
    "objaverse": [
        # radial
        ('015c200ce786438c8e35ddf635d1e236', 4),
        ('02ca74ef6a1b4ec386c11048603f0e98', 4),
        ('03c0260373c7406ea408c2dec9f8d502', 4),
        ('047ca62b79d140d4b64044db311561d9', 8),
        # spherical
        ('035d9ce9964b42f6bc20514853934d1b', 0),
        ('0050f76a07fa43b7a38e6cef40beb69d', 2),
        ('03e4ff99ecc24e5f8ea49d4e8df876e8', 2),
        ('006373e3885b472cb5538fc570235fcf', 8),
        ('002aec05c41342dea61828f67d340d2d', 8)
    ],
    "co3d_seen": [
        # 2 views for figure
        # ('191_20615_38267', 2),
        # ('256_27676_55062', 1),
        # ('212_22443_46420', 0),
        # ('12_104_640', 1),
        # ('118_13853_28129', 0),
        # ('184_19904_39401', 1),
        # ('256_27647_53803', 0),
        # ('223_23574_49752', 0),
        # ('201_21573_40063', 0),

        # 4 views for video
        ('256_27676_55062', 2),
        ('118_13853_28129', 0),
        ('112_13308_25702', 0),
        ('187_20181_35751', 2),
        ('164_17988_33493', 3),
        ('250_26773_54489', 1),
        ('222_23409_48576', 3),

    ],
}

# Visual constants
HEADER_TEXT_HEIGHT = 32
SCENE_LABEL_WIDTH = 120
VIEW_IDX_WIDTH = 50
COLUMN_GAP = 30
TEXT_COLOR = (0, 0, 0)
WARNING_COLOR = (200, 0, 0)
LABEL_HEIGHT = 30
OUTPUT_FPS = 12
SPACER_WIDTH = 20
OBJV_RADIAL_INDEX = "/home/yuwu3/prope/assets/objaverse_index_test_context2_radial.json"
OBJV_SPH_INDEX = "/home/yuwu3/prope/assets/objaverse_index_test_context2_spherical.json"


def get_log_dir(log_root, model_dir, dataset, exp_name):
    """Construct log directory path for an experiment."""
    return os.path.join(log_root, model_dir, dataset, "unknown_d", exp_name)


def get_tests_subdir(input_views):
    """Get the tests subdirectory name based on input views."""
    if input_views == 2:
        return 'tests'
    else:
        return f'eval-context{input_views}'


def load_image(path):
    """Load an image from path, return None if not found."""
    if os.path.exists(path):
        return Image.open(path).convert('RGB')
    print(f"Warning: {path} not found")
    return None


def load_image_tensor(path, device='cuda'):
    """Load image as a tensor normalized to [0, 1]."""
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('RGB')
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0).to(device)


def compute_psnr(pred_path, gt_path, device='cuda'):
    """Compute PSNR between predicted and GT images."""
    pred = load_image_tensor(pred_path, device)
    gt = load_image_tensor(gt_path, device)
    if pred is None or gt is None:
        return None
    return peak_signal_noise_ratio(pred, gt, data_range=1.0).item()


def compute_lpips(pred_path, gt_path, lpips_fn, device='cuda'):
    """Compute LPIPS between predicted and GT images."""
    pred = load_image_tensor(pred_path, device)
    gt = load_image_tensor(gt_path, device)
    if pred is None or gt is None:
        return None
    return lpips_fn(pred, gt).item()


def draw_centered_text(canvas, box, text, target_height, color):
    """Draw text centered within a bounding box."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/google-droid/DroidSans.ttf", 40)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    x0, y0, x1, y1 = box
    pos_x = x0 + (x1 - x0 - text_w) // 2
    pos_y = y0 + (y1 - y0 - text_h) // 2
    
    draw.text((pos_x, pos_y), text, fill=color, font=font)


def draw_horizontal_text(canvas, box, text, color):
    """Draw text horizontally centered within a bounding box.
    Auto-scales font size to fit the entire text within the box."""
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0
    
    # Start with a reasonable font size and decrease until text fits
    font_size = 14
    min_font_size = 6
    font = None
    text_w, text_h = 0, 0
    
    while font_size >= min_font_size:
        try:
            font = ImageFont.truetype("/usr/share/fonts/google-droid/DroidSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
            break
        
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # Check if text fits in box
        if text_w + 4 <= box_w and text_h + 4 <= box_h:
            break
        font_size -= 1
    
    if font is None:
        font = ImageFont.load_default()
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    
    # Calculate centered position
    draw = ImageDraw.Draw(canvas)
    pos_x = x0 + (box_w - text_w) // 2
    pos_y = y0 + (box_h - text_h) // 2
    
    draw.text((pos_x, pos_y), text, fill=color, font=font)


def draw_vertical_text(canvas, box, text, color):
    """Draw text vertically centered (rotated 90 degrees) within a bounding box.
    Auto-scales font size to fit the entire text within the box."""
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0
    
    # Start with a reasonable font size and decrease until text fits
    font_size = 14
    min_font_size = 6
    font = None
    text_w, text_h = 0, 0
    
    while font_size >= min_font_size:
        try:
            font = ImageFont.truetype("/usr/share/fonts/google-droid/DroidSans.ttf", font_size)
        except:
            font = ImageFont.load_default()
            break
        
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
        
        # After rotation: text_w becomes height, text_h becomes width
        # Check if rotated text fits in box
        if text_h + 4 <= box_w and text_w + 4 <= box_h:
            break
        font_size -= 1
    
    if font is None:
        font = ImageFont.load_default()
        temp_draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    
    # Create text image
    text_img = Image.new('RGB', (text_w + 4, text_h + 4), 'white')
    text_draw = ImageDraw.Draw(text_img)
    text_draw.text((2, 2), text, fill=color, font=font)
    
    # Rotate 90 degrees counter-clockwise
    text_img = text_img.rotate(90, expand=True)
    
    # Calculate position
    rotated_w, rotated_h = text_img.size
    
    paste_x = x0 + (box_w - rotated_w) // 2
    paste_y = y0 + (box_h - rotated_h) // 2
    
    canvas.paste(text_img, (paste_x, paste_y))


def eval_metric_advantage_fn(csv_file, log_root, model_dir, dataset, main_comp_exps, input_views=2):
    '''
    For PRoPE and RayRoPE, load generated views and GT views, compute PSNR and LPIPS.
    For each view in each scene, (a scene may have multiple views),
    compute the advantage of RayRoPE over PRoPE in terms of these metrics and save to a .csv file
    with columns: 'scene', 'view_id', 'psnr_diff', 'lpips_diff'
    
    psnr_diff = RayRoPE_psnr - PRoPE_psnr (positive means RayRoPE is better)
    lpips_diff = PRoPE_lpips - RayRoPE_lpips (positive means RayRoPE is better, since lower LPIPS is better)
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(device)
    
    prope_dir = get_log_dir(log_root, model_dir, dataset, main_comp_exps['PRoPE'])
    rayrope_dir = get_log_dir(log_root, model_dir, dataset, main_comp_exps['RayRoPE'])
    
    tests_subdir = get_tests_subdir(input_views)
    prope_tests_dir = os.path.join(prope_dir, tests_subdir)
    if not os.path.exists(prope_tests_dir):
        print(f"Error: PRoPE tests directory not found: {prope_tests_dir}")
        return
    
    scenes = [d for d in os.listdir(prope_tests_dir) if os.path.isdir(os.path.join(prope_tests_dir, d))]
    
    results = []
    
    for scene in tqdm(scenes, desc="Computing metrics"):
        prope_scene_dir = os.path.join(prope_tests_dir, scene)
        rayrope_scene_dir = os.path.join(rayrope_dir, tests_subdir, scene)
        
        if not os.path.exists(rayrope_scene_dir):
            continue
        
        # Find all generated views
        gen_files = [f for f in os.listdir(prope_scene_dir) if f.startswith('gen') and f.endswith('.png')]
        
        for gen_file in gen_files:
            view_id = gen_file.replace('gen', '').replace('.png', '')
            
            prope_gen_path = os.path.join(prope_scene_dir, f'gen{view_id}.png')
            rayrope_gen_path = os.path.join(rayrope_scene_dir, f'gen{view_id}.png')
            gt_path = os.path.join(prope_scene_dir, f'target{view_id}.png')
            
            if not all(os.path.exists(p) for p in [prope_gen_path, rayrope_gen_path, gt_path]):
                continue
            
            # Compute PSNR
            prope_psnr = compute_psnr(prope_gen_path, gt_path, device)
            rayrope_psnr = compute_psnr(rayrope_gen_path, gt_path, device)
            
            # Compute LPIPS
            prope_lpips = compute_lpips(prope_gen_path, gt_path, lpips_fn, device)
            rayrope_lpips = compute_lpips(rayrope_gen_path, gt_path, lpips_fn, device)
            
            if all(v is not None for v in [prope_psnr, rayrope_psnr, prope_lpips, rayrope_lpips]):
                psnr_diff = rayrope_psnr - prope_psnr  # positive = RayRoPE better
                lpips_diff = prope_lpips - rayrope_lpips  # positive = RayRoPE better
                
                results.append({
                    'scene': scene,
                    'view_id': view_id,
                    'psnr_diff': psnr_diff,
                    'lpips_diff': lpips_diff,
                    'prope_psnr': prope_psnr,
                    'rayrope_psnr': rayrope_psnr,
                    'prope_lpips': prope_lpips,
                    'rayrope_lpips': rayrope_lpips,
                })
    
    # Save to CSV
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['scene', 'view_id', 'psnr_diff', 'lpips_diff', 
                                                'prope_psnr', 'rayrope_psnr', 'prope_lpips', 'rayrope_lpips'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Saved metrics to {csv_file} ({len(results)} entries)")


def get_objv_partition_indices(scene, partition):
    """
    Get the valid view indices for a given objaverse scene and partition.
    
    Args:
        scene: Scene name/id
        partition: 'radial' or 'spherical'
    
    Returns:
        Set of valid view indices (as strings) for this partition, or None if scene not found.
    """
    if partition not in ['radial', 'spherical']:
        return None
    
    # Load the partition-specific index
    partition_index_file = OBJV_RADIAL_INDEX if partition == 'radial' else OBJV_SPH_INDEX
    all_index_file = "/home/yuwu3/prope/assets/objaverse_index_test_context2_all.json"
    
    if not os.path.exists(partition_index_file) or not os.path.exists(all_index_file):
        return None
    
    with open(partition_index_file, 'r') as f:
        partition_data = json.load(f)
    with open(all_index_file, 'r') as f:
        all_data = json.load(f)
    
    if scene not in partition_data or scene not in all_data:
        return None
    
    partition_targets = partition_data[scene]['target_view_files']
    all_targets = all_data[scene]['target_view_files']
    
    # Find indices in the full list that correspond to the partition targets
    valid_indices = set()
    for target in partition_targets:
        if target in all_targets:
            idx = all_targets.index(target)
            valid_indices.add(str(idx))
    
    return valid_indices


def get_rank_fn(csv_file, lpips_w=0.0, dataset=None, objv_partition=None):
    '''
    Load from the csv file generated by eval_metric_advantage_fn(),
    rank from most advantage to least advantage based on psnr and lpips separately.
    The final rank is the weighted sum of the two ranks: 
    final_rank = lpips_w * lpips_rank + (1 - lpips_w) * psnr_rank
    return the sorted scene list based on the final rank (list of (scene, view_id) tuples)
    
    Args:
        csv_file: Path to CSV file with metric advantages
        lpips_w: Weight for LPIPS in ranking (0.0 = PSNR only, 1.0 = LPIPS only)
        dataset: Dataset name (used for objaverse partition filtering)
        objv_partition: For objaverse, filter to 'radial' or 'spherical' views only
    '''
    if not os.path.exists(csv_file):
        print(f"Error: CSV file not found: {csv_file}")
        return []
    
    entries = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                'scene': row['scene'],
                'view_id': row['view_id'],
                'psnr_diff': float(row['psnr_diff']),
                'lpips_diff': float(row['lpips_diff']),
            })
    
    if not entries:
        return []
    
    # Filter for objaverse partition if specified
    if dataset == 'objaverse' and objv_partition in ['radial', 'spherical']:
        filtered_entries = []
        partition_cache = {}  # Cache valid indices per scene
        
        for e in entries:
            scene = e['scene']
            if scene not in partition_cache:
                partition_cache[scene] = get_objv_partition_indices(scene, objv_partition)
            
            valid_indices = partition_cache[scene]
            if valid_indices is not None and e['view_id'] in valid_indices:
                filtered_entries.append(e)
        
        entries = filtered_entries
        print(f"Filtered to {len(entries)} entries for {objv_partition} partition")
    
    if not entries:
        return []
    
    # For each scene, pick the view with the highest advantage score
    # Advantage score = lpips_w * lpips_diff + (1 - lpips_w) * psnr_diff
    scene_best = {}
    for e in entries:
        scene = e['scene']
        advantage_score = lpips_w * e['lpips_diff'] + (1 - lpips_w) * e['psnr_diff']
        
        if scene not in scene_best or advantage_score > scene_best[scene][1]:
            scene_best[scene] = (e['view_id'], advantage_score)
    
    # Convert to list and sort by advantage score (higher is better)
    scene_list = [(scene, view_id, score) for scene, (view_id, score) in scene_best.items()]
    scene_list.sort(key=lambda x: x[2], reverse=True)
    
    return [(scene, view_id) for scene, view_id, _ in scene_list]


def create_img_comp_fn(scene_list, img_out_file, log_root, model_dir, dataset, main_comp_exps, input_views=2, show_scene_label=True, show_view_idx=True):
    """
    Create a comparison image visualization for the given scenes.
    Layout: [Scene Label |] [View Idx |] Ref0 | Ref1 | ... | [gap] | GT | Plucker | GTA | PRoPE | RayRoPE
    Each row is one scene/view.
    
    Args:
        input_views: Number of input/reference views.
        show_scene_label: If True, display scene name as horizontal label on the left.
        show_view_idx: If True, display view index as horizontal label after scene label.
    """
    if not scene_list:
        print("No scenes to visualize")
        return
    
    # Get log directories
    method_dirs = {}
    for method, exp_name in main_comp_exps.items():
        method_dirs[method] = get_log_dir(log_root, model_dir, dataset, exp_name)
    
    tests_subdir = get_tests_subdir(input_views)
    
    # Use PRoPE dir to get sample image size
    first_scene, first_view_id = scene_list[0]
    sample_path = os.path.join(method_dirs['PRoPE'], tests_subdir, first_scene, 'ref0.png')
    sample_img = load_image(sample_path)
    if sample_img is None:
        print(f"Could not load sample image: {sample_path}")
        return
    
    img_w, img_h = sample_img.size
    
    # Column labels: [Scene |] [View |] Ref0 | Ref1 | ... | [gap] | GT | methods...
    method_labels = ['Plucker', 'GTA', 'PRoPE', 'RayRoPE']
    ref_labels = [f'Ref View {i}' for i in range(input_views)]
    labels = ref_labels + ['GT Target'] + method_labels
    
    # Calculate canvas size
    n_content_cols = len(labels)  # columns for images (excluding scene label and view idx)
    header_h = max(int(HEADER_TEXT_HEIGHT * 1.6), HEADER_TEXT_HEIGHT + 20)
    scene_label_w = SCENE_LABEL_WIDTH if show_scene_label else 0
    view_idx_w = VIEW_IDX_WIDTH if show_view_idx else 0
    total_w = scene_label_w + view_idx_w + img_w * input_views + COLUMN_GAP + img_w * (1 + len(method_labels))
    total_h = header_h + len(scene_list) * img_h
    
    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Draw header labels
    x = scene_label_w + view_idx_w
    for idx, label in enumerate(labels):
        if idx == 2:  # Add gap before GT
            x += COLUMN_GAP
        box = (x, 0, x + img_w, header_h)
        draw_centered_text(canvas, box, label, HEADER_TEXT_HEIGHT, TEXT_COLOR)
        x += img_w
    
    # Draw scene name header if showing scene labels
    if show_scene_label:
        box = (0, 0, SCENE_LABEL_WIDTH, header_h)
        draw_centered_text(canvas, box, "Scene", HEADER_TEXT_HEIGHT, TEXT_COLOR)
    
    # Draw view index header if showing view index
    if show_view_idx:
        box = (scene_label_w, 0, scene_label_w + VIEW_IDX_WIDTH, header_h)
        draw_centered_text(canvas, box, "View", HEADER_TEXT_HEIGHT, TEXT_COLOR)
    
    # Draw each row
    for row, (scene, view_id) in enumerate(scene_list):
        y = header_h + row * img_h
        
        # Draw scene label (full scene name, horizontal text auto-scaled to fit)
        if show_scene_label:
            scene_box = (0, y, SCENE_LABEL_WIDTH, y + img_h)
            draw_horizontal_text(canvas, scene_box, scene, TEXT_COLOR)
        
        # Draw view index (horizontal text)
        if show_view_idx:
            view_box = (scene_label_w, y, scene_label_w + VIEW_IDX_WIDTH, y + img_h)
            draw_horizontal_text(canvas, view_box, str(view_id), TEXT_COLOR)
        
        x = scene_label_w + view_idx_w
        
        # Load all reference views
        for ref_idx in range(input_views):
            ref_path = os.path.join(method_dirs['PRoPE'], tests_subdir, scene, f'ref{ref_idx}.png')
            img = load_image(ref_path)
            if img:
                if img.size != (img_w, img_h):
                    img = img.resize((img_w, img_h), Image.LANCZOS)
                canvas.paste(img, (x, y))
            else:
                draw.rectangle((x, y, x + img_w - 1, y + img_h - 1), outline='red', width=2)
            x += img_w
        
        # Gap
        x += COLUMN_GAP
        
        # GT
        gt_path = os.path.join(method_dirs['PRoPE'], tests_subdir, scene, f'target{view_id}.png')
        img = load_image(gt_path)
        if img:
            if img.size != (img_w, img_h):
                img = img.resize((img_w, img_h), Image.LANCZOS)
            canvas.paste(img, (x, y))
        else:
            draw.rectangle((x, y, x + img_w - 1, y + img_h - 1), outline='red', width=2)
        x += img_w
        
        # Method outputs
        for method in method_labels:
            gen_path = os.path.join(method_dirs[method], tests_subdir, scene, f'gen{view_id}.png')
            img = load_image(gen_path)
            if img:
                if img.size != (img_w, img_h):
                    img = img.resize((img_w, img_h), Image.LANCZOS)
                canvas.paste(img, (x, y))
            else:
                draw.rectangle((x, y, x + img_w - 1, y + img_h - 1), outline='red', width=2)
                draw_centered_text(canvas, (x, y, x + img_w, y + img_h), "Not found", 16, WARNING_COLOR)
            x += img_w
    
    os.makedirs(os.path.dirname(img_out_file), exist_ok=True)
    canvas.save(img_out_file, quality=95)
    print(f"Saved comparison image to {img_out_file}")
    print(f"Visualized {len(scene_list)} scenes:")
    for scene, view_id in scene_list:
        print(f"  Scene: {scene}, View: {view_id}")


def add_label(frame, text):
    """Add a label at the top of a frame."""
    h, w = frame.shape[:2]
    labeled = np.ones((h + LABEL_HEIGHT, w, 3), dtype=np.uint8) * 255
    labeled[LABEL_HEIGHT:] = frame
    
    # Add text using PIL for better rendering
    pil_img = Image.fromarray(labeled)
    draw = ImageDraw.Draw(pil_img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSansMono-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = (w - text_w) // 2
    text_y = (LABEL_HEIGHT - (bbox[3] - bbox[1])) // 2
    
    draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    return np.array(pil_img)


def add_scene_label(frame, text, show_label=True):
    """Add a scene label to the left of a frame.
    
    Args:
        frame: The frame to add label to
        text: The scene name text
        show_label: If True, add the label; if False, return frame unchanged
    """
    if not show_label:
        return frame
    
    h, w = frame.shape[:2]
    labeled = np.ones((h, w + SCENE_LABEL_WIDTH, 3), dtype=np.uint8) * 255
    labeled[:, SCENE_LABEL_WIDTH:] = frame
    
    pil_img = Image.fromarray(labeled)
    # Display full scene name (auto-scaled by draw_vertical_text)
    draw_vertical_text(pil_img, (0, 0, SCENE_LABEL_WIDTH, h), text, TEXT_COLOR)
    
    return np.array(pil_img)


def load_video(path):
    """Load video frames from path."""
    if not os.path.exists(path):
        print(f"Warning: {path} not found")
        return None
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def create_video_comp_fn(scene_list, video_out_file, log_root, model_dir, dataset, main_comp_exps, input_views=2, show_scene_label=True):
    """
    Create a comparison video visualization for the given scenes.
    Layout for each frame: [Scene Label |] Ref Views | [gap] | [GT |] Plucker | GTA | PRoPE | RayRoPE
    
    Args:
        input_views: Number of input/reference views.
        show_scene_label: If True, display scene name as vertical label on the left.
    
    Note: GT is not visualized for objaverse and co3d datasets.
    """
    if not scene_list:
        print("No scenes to visualize")
        return
    
    # Determine whether to show GT (not for objaverse or co3d)
    # show_gt = dataset not in ['objaverse', 'co3d_seen']
    show_gt = True
    
    # Get log directories
    method_dirs = {}
    for method, exp_name in main_comp_exps.items():
        method_dirs[method] = get_log_dir(log_root, model_dir, dataset, exp_name)
    
    tests_subdir = get_tests_subdir(input_views)
    
    method_labels = ['Plucker', 'GTA', 'PRoPE', 'RayRoPE']
    all_frames = []
    frame_size = None
    scene_label_w = SCENE_LABEL_WIDTH if show_scene_label else 0
    
    # Get unique scenes (each scene may have multiple views, but video is per scene)
    unique_scenes = list(dict.fromkeys([scene for scene, _ in scene_list]))
    
    for scene in tqdm(unique_scenes, desc="Processing videos"):
        # Load GT video (from PRoPE dir)
        gt_video_path = os.path.join(method_dirs['PRoPE'], tests_subdir, scene, 'gt.mp4')
        gt_frames = load_video(gt_video_path)
        
        if gt_frames is None or len(gt_frames) == 0:
            print(f"Skipping scene {scene}: GT video not found or empty")
            continue
        
        # Load method videos
        method_frames = {'GT': gt_frames}
        skip_scene = False
        for method in method_labels:
            video_path = os.path.join(method_dirs[method], tests_subdir, scene, 'pred.mp4')
            frames = load_video(video_path)
            if frames is None or len(frames) == 0:
                print(f"Skipping scene {scene}: {method} video not found")
                skip_scene = True
                break
            method_frames[method] = frames
        
        if skip_scene:
            continue
        
        # Load all reference images
        ref_imgs = []
        refs_valid = True
        for ref_idx in range(input_views):
            ref_path = os.path.join(method_dirs['PRoPE'], tests_subdir, scene, f'ref{ref_idx}.png')
            ref_img = load_image(ref_path)
            if ref_img is None:
                print(f"Skipping scene {scene}: ref{ref_idx}.png not found")
                refs_valid = False
                break
            ref_imgs.append(ref_img)
        
        if not refs_valid:
            continue
        
        # Get frame dimensions
        h, w = gt_frames[0].shape[:2]
        half_w = w // 2
        half_h = h // 2
        
        # Calculate ref view grid layout
        # Ref views are half height/width of video, arranged in a grid
        # Number of columns = ceil(input_views / 2), 2 rows
        num_ref_cols = (input_views + 1) // 2  # ceil division
        num_ref_rows = 2 if input_views > 1 else 1
        ref_column_w = half_w * num_ref_cols
        
        # Compute frame size if not set
        if frame_size is None:
            # Scene label + ref_column_w (ref grid) + spacer + w * (num_videos)
            # num_videos = 1 GT (if show_gt) + 4 methods
            num_videos = (1 if show_gt else 0) + len(method_labels)
            frame_w = scene_label_w + ref_column_w + SPACER_WIDTH + w * num_videos
            frame_h = h + LABEL_HEIGHT
            frame_size = (frame_w, frame_h)
        
        # Resize reference views to half size
        ref_resized = [np.array(img.resize((half_w, half_h), Image.LANCZOS)) for img in ref_imgs]
        
        # Arrange refs in a grid: num_ref_cols columns x 2 rows
        # First row: refs 0, 2, 4, ... (even indices)
        # Second row: refs 1, 3, 5, ... (odd indices)
        row1_refs = [ref_resized[i] for i in range(0, input_views, 2)]
        row2_refs = [ref_resized[i] for i in range(1, input_views, 2)]
        
        # Pad rows if needed to have equal number of columns
        white_tile = np.ones((half_h, half_w, 3), dtype=np.uint8) * 255
        while len(row1_refs) < num_ref_cols:
            row1_refs.append(white_tile)
        while len(row2_refs) < num_ref_cols:
            row2_refs.append(white_tile)
        
        # Stack horizontally to form rows
        ref_row1 = np.concatenate(row1_refs, axis=1)
        ref_row2 = np.concatenate(row2_refs, axis=1)
        
        # Add label to first row
        ref_row1_labeled = add_label(ref_row1, "Ref views")
        
        # Stack vertically
        ref_column_raw = np.vstack([ref_row1_labeled, ref_row2])
        
        # Target height should match video row height (h + LABEL_HEIGHT)
        target_height = h + LABEL_HEIGHT
        ref_col_h, ref_col_w = ref_column_raw.shape[:2]
        
        if ref_col_h < target_height:
            # Pad with white at the bottom
            pad_height = target_height - ref_col_h
            padding = np.ones((pad_height, ref_col_w, 3), dtype=np.uint8) * 255
            ref_column = np.vstack([ref_column_raw, padding])
        elif ref_col_h > target_height:
            # Crop from the bottom (should rarely happen with proper sizing)
            ref_column = ref_column_raw[:target_height, :, :]
        else:
            ref_column = ref_column_raw
        
        # Create white spacer
        spacer = np.ones((target_height, SPACER_WIDTH, 3), dtype=np.uint8) * 255
        
        # Find minimum frame count across videos (exclude GT if not showing)
        videos_to_check = (['GT'] if show_gt else []) + method_labels
        min_frames = min(len(method_frames[label]) for label in videos_to_check)
        
        scene_frames = []
        for i in range(min_frames):
            # Add labels to video frames
            labeled_frames = []
            # Include GT only if show_gt is True
            video_labels = (['GT'] if show_gt else []) + method_labels
            for label in video_labels:
                labeled_frame = add_label(method_frames[label][i], label)
                labeled_frames.append(labeled_frame)
            
            video_row = np.concatenate(labeled_frames, axis=1)
            
            # Combine: ref_column + spacer + video_row
            row = np.concatenate([ref_column, spacer, video_row], axis=1)
            
            # Add scene label on the left (if enabled)
            row_with_scene = add_scene_label(row, scene, show_label=show_scene_label)
            
            scene_frames.append(row_with_scene)
        
        # Add frames forward and backward
        all_frames.extend(scene_frames)
        all_frames.extend(scene_frames[::-1])
    
    if not all_frames:
        print("No frames to write")
        return
    
    # Verify frame size
    actual_h, actual_w = all_frames[0].shape[:2]
    if (actual_w, actual_h) != frame_size:
        print(f"Warning: Expected frame size {frame_size}, got ({actual_w}, {actual_h})")
        frame_size = (actual_w, actual_h)
    
    # Write video
    os.makedirs(os.path.dirname(video_out_file), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_out_file, fourcc, OUTPUT_FPS, frame_size)
    
    if not out.isOpened():
        print("Error: Could not open video writer")
        return
    
    for frame in all_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"Saved video to {video_out_file} ({len(all_frames)} frames, size={frame_size})")


if __name__ == "__main__":
    do_eval_metric_advantage = False
    do_create_img_comp = False
    do_create_video_comp = True
    
    log_root = "/grogu/user/yuwu3/rayrope_log_Dec/"
    model_dir = "L6-H8-D1152-FF1024-B8"
    input_views = 4 # 2 is default
    dataset = "co3d_seen"  # "re10k" or "objaverse" or "co3d_seen"
    objv_partition = "spherical" # "radial" or "spherical"
    show_scene_label = True
    show_view_idx = False

    main_comp_exps = {
        'Plucker': 'none-plucker-seed1',
        # 'Plucker': 'masked/none-plucker-pw0.01-seed1', # for objaverse
        'GTA': 'gta-none-seed1',
        'PRoPE': 'prope-seed1',
        'RayRoPE': 'd_pj+0_3d-predict_dsig-inv_d-seed1', 
    }
    # log dir for each exp has format: {log_root}/{model_dir}/{dataset}/unknown_d/{exp_name}

    # Add input_views suffix to file names
    views_suffix = f"_ctx{input_views}" if input_views != 2 else ""
    
    csv_file = f"./visual_comp/{dataset}{views_suffix}_metric_advantage.csv"
    if do_eval_metric_advantage:
        eval_metric_advantage_fn(csv_file, log_root, model_dir, dataset, main_comp_exps, input_views=input_views)
    
    FINAL_PICKED = True
    if FINAL_PICKED:
        scene_list = PICKED_SCENES[dataset]
        img_out_file = f"./visual_comp/{dataset}{views_suffix}_final.png"
        video_out_file = f"./visual_comp/{dataset}{views_suffix}_final.mp4"
    else:
        lpips_w = 1.0
        top_n = 20
        scene_list = get_rank_fn(csv_file, lpips_w=lpips_w, dataset=dataset, objv_partition=objv_partition)
        scene_list = scene_list[:top_n]
        
        # Add partition suffix for objaverse
        partition_suffix = f"_{objv_partition}" if dataset == 'objaverse' and objv_partition else ""
        img_out_file = f"./visual_comp/{dataset}{views_suffix}_lpipsw_{lpips_w}{partition_suffix}.png"
        video_out_file = f"./visual_comp/{dataset}{views_suffix}_lpipsw_{lpips_w}{partition_suffix}.mp4"

    if do_create_img_comp:
        create_img_comp_fn(scene_list, img_out_file, log_root, model_dir, dataset, main_comp_exps, input_views=input_views, show_scene_label=show_scene_label, show_view_idx=show_view_idx)

    if do_create_video_comp:
        create_video_comp_fn(scene_list, video_out_file, log_root, model_dir, dataset, main_comp_exps, input_views=input_views, show_scene_label=show_scene_label)