'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function stitch_background() and panorama().
3. If you want to show an image for debugging, please use show_image() function in util.py. 
4. Please do NOT save any intermediate files in your final submission.
'''
import torch
import kornia as K
from typing import Dict
from utils import show_image
'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''
# ------------------------------------ Task 1 ------------------------------------ #
def stitch_background(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: input images are a dict of 2 images of torch.Tensor represent an input images for task-1.
    Returns:
        img: stitched_image: torch.Tensor of the output image.
    """
    #TODO: Add your code here. Do not modify the return and input arguments.

    keys = sorted(imgs.keys())
    img1 = imgs[keys[0]].float() / 255.0
    img2 = imgs[keys[1]].float() / 255.0

    gray1 = K.color.rgb_to_grayscale(img1).unsqueeze(0)
    gray2 = K.color.rgb_to_grayscale(img2).unsqueeze(0)

    sift = K.feature.SIFTFeature()
    lafs1, _, desc1 = sift(gray1)
    lafs2, _, desc2 = sift(gray2)
    kps1 = K.feature.get_laf_center(lafs1)[0]
    kps2 = K.feature.get_laf_center(lafs2)[0]

    matcher = K.feature.DescriptorMatcher(match_mode='snn', th=0.75)
    _, idx = matcher(desc1[0], desc2[0])
    matched_kp1 = kps1[idx[:, 0].long()]
    matched_kp2 = kps2[idx[:, 1].long()]

    ransac = K.geometry.ransac.RANSAC(
        model_type='homography', inl_th=1.0, max_iter=2500, confidence=0.99
    )
    H_1to2, _ = ransac(matched_kp1, matched_kp2)
    H_1to2 = H_1to2.squeeze(0)

    C, h1, w1 = img1.shape
    _, h2, w2 = img2.shape

    H_list = [H_1to2, torch.eye(3, dtype=torch.float32)]
    img_shapes = [(h1, w1), (h2, w2)]
    out_h, out_w, T = compute_output_canvas(H_list, img_shapes)

    H1_final = T @ H_1to2
    H2_final = T

    warped1 = K.geometry.warp_perspective(
        img1.unsqueeze(0), H1_final.unsqueeze(0), dsize=(out_h, out_w)
    )[0]
    warped2 = K.geometry.warp_perspective(
        img2.unsqueeze(0), H2_final.unsqueeze(0), dsize=(out_h, out_w)
    )[0]

    mask1 = (warped1.sum(dim=0) > 0).float()
    mask2 = (warped2.sum(dim=0) > 0).float()
    overlap_mask = mask1 * mask2
    only1 = mask1 * (1.0 - mask2)
    only2 = mask2 * (1.0 - mask1)

    diff = torch.norm(warped1 - warped2, dim=0) * overlap_mask
    moving_mask = (diff > 0.05).float() * overlap_mask

    kernel = torch.ones(7, 7)
    moving_mask = K.morphology.dilation(
        moving_mask.unsqueeze(0).unsqueeze(0), kernel
    )[0, 0]
    moving_mask = moving_mask.clamp(0, 1) * overlap_mask

    static_mask = overlap_mask - moving_mask

    output = torch.zeros_like(warped1)
    output += warped1 * only1.unsqueeze(0)
    output += warped2 * only2.unsqueeze(0)
    output += (warped1 + warped2) / 2.0 * static_mask.unsqueeze(0)
    output += torch.max(warped1, warped2) * moving_mask.unsqueeze(0)

    img = (output * 255).clamp(0, 255).to(torch.uint8)

    return img


def compute_output_canvas(H_list, img_shapes):
    all_corners = []
    for i, (h, w) in enumerate(img_shapes):
        corners = torch.tensor([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                               dtype=torch.float32)
        corners_h = K.geometry.convert_points_to_homogeneous(corners)
        warped = (H_list[i] @ corners_h.T).T
        warped = K.geometry.convert_points_from_homogeneous(warped)
        all_corners.append(warped)

    all_corners = torch.cat(all_corners, dim=0).detach()
    mins = all_corners.min(dim=0).values
    maxs = all_corners.max(dim=0).values

    T = torch.tensor([
        [1, 0, -mins[0].item()],
        [0, 1, -mins[1].item()],
        [0, 0, 1]
    ], dtype=torch.float32)

    out_w = int((maxs[0] - mins[0]).item()) + 2
    out_h = int((maxs[1] - mins[1]).item()) + 2

    return out_h, out_w, T


# ------------------------------------ Task 2 ------------------------------------ #
def panorama(imgs: Dict[str, torch.Tensor]):
    """
    Args:
        imgs: dict {filename: CxHxW tensor} for task-2.
    Returns:
        img: panorama, 
        overlap: torch.Tensor of the output image. 
    """
    #TODO: Add your code here. Do not modify the return and input arguments.

    file_list = sorted(imgs.keys())
    num_images = len(file_list)

    if num_images == 0:
        return torch.zeros((3, 256, 256), dtype=torch.uint8), torch.empty((0, 0))
    if num_images == 1:
        return imgs[file_list[0]], torch.zeros((1, 1))

    sift = K.feature.SIFTFeature()
    image_data = {}
    for idx, fname in enumerate(file_list):
        color_img = imgs[fname].float() / 255.0
        gray_img = K.color.rgb_to_grayscale(color_img.unsqueeze(0))
        lafs, _, descs = sift(gray_img)
        kps = K.feature.get_laf_center(lafs)[0]
        image_data[idx] = {
            'color': color_img,
            'kps': kps,
            'descs': descs[0],
        }

    overlap = torch.zeros(num_images, num_images)
    transform_dict = {}

    ransac = K.geometry.ransac.RANSAC(
        model_type='homography', inl_th=1.0, max_iter=2500, confidence=0.99
    )
    matcher = K.feature.DescriptorMatcher(match_mode='snn', th=0.75)

    min_match_count = 20
    min_inlier_count = 15

    for src_idx in range(num_images):
        for dst_idx in range(src_idx + 1, num_images):
            _, match_idx = matcher(
                image_data[src_idx]['descs'],
                image_data[dst_idx]['descs']
            )
            if match_idx.shape[0] < min_match_count:
                continue

            src_kps = image_data[src_idx]['kps'][match_idx[:, 0].long()]
            dst_kps = image_data[dst_idx]['kps'][match_idx[:, 1].long()]

            try:
                H_fwd, fwd_mask = ransac(src_kps, dst_kps)
                n_fwd = int(fwd_mask.sum().item())
            except Exception:
                H_fwd, n_fwd = None, 0

            try:
                H_rev, rev_mask = ransac(dst_kps, src_kps)
                n_rev = int(rev_mask.sum().item())
            except Exception:
                H_rev, n_rev = None, 0

            best_inliers = max(n_fwd, n_rev)
            if best_inliers < min_inlier_count:
                continue

            overlap[src_idx, dst_idx] = 1.0
            overlap[dst_idx, src_idx] = 1.0

            if n_fwd >= n_rev:
                H_s2d = H_fwd.squeeze(0)
                H_d2s = torch.linalg.inv(H_s2d)
            else:
                H_d2s = H_rev.squeeze(0)
                H_s2d = torch.linalg.inv(H_d2s)

            transform_dict[(src_idx, dst_idx)] = H_s2d
            transform_dict[(dst_idx, src_idx)] = H_d2s

    connect_degrees = overlap.sum(dim=0)
    reference_idx = int(connect_degrees.argmax().item())

    transforms_to_ref = bfs_homographies(
        num_images, transform_dict, overlap, reference_idx
    )

    for i in range(num_images):
        if i not in transforms_to_ref:
            transforms_to_ref[i] = torch.eye(3, dtype=torch.float32)

    H_list = [transforms_to_ref[i] for i in range(num_images)]
    img_shapes = [(image_data[i]['color'].shape[1], image_data[i]['color'].shape[2])
                  for i in range(num_images)]
    out_h, out_w, T = compute_output_canvas(H_list, img_shapes)

    warped_imgs = []
    masks = []
    weights = []
    for i in range(num_images):
        H_final = T @ transforms_to_ref[i]
        src_img = image_data[i]['color'].unsqueeze(0)
        warped, mask, weight = warp_image_float(src_img, H_final, out_h, out_w)
        warped_imgs.append(warped)
        masks.append(mask)
        weights.append(weight)

    result = torch.zeros(3, out_h, out_w, dtype=torch.float32)
    total_weight = torch.zeros(out_h, out_w, dtype=torch.float32)

    for i in range(num_images):
        w = weights[i]
        result += warped_imgs[i] * w.unsqueeze(0)
        total_weight += w

    total_weight = total_weight.clamp(min=1e-8)
    result = result / total_weight.unsqueeze(0)

    img = (result * 255).clamp(0, 255).to(torch.uint8)

    return img, overlap


def bfs_homographies(n, transform_dict, overlap_matrix, ref_idx):
    conn_graph = {node: [] for node in range(n)}
    for src in range(n):
        for dst in range(src + 1, n):
            if overlap_matrix[src, dst] > 0:
                conn_graph[src].append(dst)
                conn_graph[dst].append(src)

    transforms_to_ref = {ref_idx: torch.eye(3, dtype=torch.float32)}
    visited = {ref_idx}
    queue = [ref_idx]

    while queue:
        current = queue.pop(0)
        for neighbor in conn_graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                H_n_to_c = transform_dict[(neighbor, current)]
                transforms_to_ref[neighbor] = transforms_to_ref[current] @ H_n_to_c

    return transforms_to_ref


def make_center_weight(h, w):
    wy = torch.minimum(torch.arange(h, dtype=torch.float32),
                        torch.arange(h - 1, -1, -1, dtype=torch.float32)) + 1.0
    wx = torch.minimum(torch.arange(w, dtype=torch.float32),
                        torch.arange(w - 1, -1, -1, dtype=torch.float32)) + 1.0
    weight = torch.minimum(wy.unsqueeze(1).expand(h, w),
                           wx.unsqueeze(0).expand(h, w))
    return weight.unsqueeze(0).unsqueeze(0)


def warp_image_float(img_f, H, out_h, out_w):
    warped = K.geometry.warp_perspective(
        img_f, H.unsqueeze(0), dsize=(out_h, out_w)
    )[0]

    _, _, sh, sw = img_f.shape
    ones = torch.ones(1, 1, sh, sw, dtype=torch.float32)
    mask_w = K.geometry.warp_perspective(
        ones, H.unsqueeze(0), dsize=(out_h, out_w)
    )[0, 0]
    binary_mask = mask_w > 0.5

    cw = make_center_weight(sh, sw)
    weight_w = K.geometry.warp_perspective(
        cw, H.unsqueeze(0), dsize=(out_h, out_w)
    )[0, 0]
    weight_w = weight_w * binary_mask.float()

    return warped, binary_mask, weight_w