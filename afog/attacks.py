from utils.attack_utils.target_utils import generate_attack_targets
import numpy as np
import torch

"""
The core AFOG attack, which handles baseline, vanishing, and fabrication modes. See our paper for formulation of AFOG and
empirical results. 

victim: model object
x_query: input image x to be attacked
n_iter: number of iterations of attack
eps: maximum perturbation budget
eps_iter: perturbation adjustment per iteration
attn_lr: attention learning rate
vis: enable visualization mode, returning extra arrays
mode: attack mode, either "baseline", "vanishing", or "fabrication"
meta: additional parameter to play nice with multiple model architectures

returns: attacked image x_adv
"""
def afog(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255., attn_lr=0.01, vis=False, mode="baseline", meta=None):
    # Select loss function variation based on mode
    attack_modes = {
        "baseline":victim.compute_object_untargeted_gradient,
        "vanishing":victim.compute_object_vanishing_gradient,
        "fabrication":victim.compute_object_fabrication_gradient,
    }
    
    # If visualization is enabled, save intermediate arrays and images
    if vis:
        perts = []
        pert_grads = []
        map_grads = []
        maps = []
    
    # Scale eps according to the range of the image ([0, 1] or [0, 255] typically)
    init_min = min(0.0, torch.min(x_query).item())
    init_max = torch.max(x_query).item()
    mult_factor = init_max

    eps_iter = eps_iter * mult_factor
    eps = eps * mult_factor
    
    # Initialize perturbation map and attn_map
    pert = np.random.uniform(-eps, eps, size=x_query.shape)
    attn_map = np.ones((x_query.shape))
       
    # Get detections and initialize pert and attn. Sometimes there is additional metadata to pass in
    detections_query = None
    if meta == None:
        detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    else:
        detections_query = victim.detect(x_query, meta=meta)
        detections_query[0]['meta'] = meta # also want to pass in meta here so that the gradient functions can access it
    
    # Make the first adversarial example
    x_adv = x_query + np.multiply(attn_map, pert)

    for i in range(n_iter):
        
        # save for visualization
        if vis:
            perts.append(pert.copy())
            maps.append(attn_map.copy())

        # first get the gradient for x_adv
        grad = attack_modes[mode](x_adv, x_query, detections = detections_query, norm=False)
        
        # compute the two partials. Simple multiplication because this is elementwise, not matmul
        pert_grad = np.multiply(grad, attn_map)
        attn_map_grad = np.multiply(grad, pert)
        
        # update perturbation
        signed_pert_grad = np.sign(pert_grad)
        pert -= eps_iter * signed_pert_grad
        pert = np.clip(pert, -eps, eps)
        
        # update attention
        norm_attn_map_grad = (attn_map_grad - np.mean(attn_map_grad)) / np.std(attn_map_grad)
        attn_map -= attn_lr * norm_attn_map_grad
        
        # save for visualization
        if vis:
            pert_grads.append(pert_grad.copy())
            map_grads.append(attn_map_grad.copy())
    
        # make the next iteration of x_adv
        pert = np.clip(np.multiply(attn_map, pert), -eps, eps)
        x_adv = np.clip(x_query + pert, init_min, init_max)
    
    # create final version of attacked image once iteration is complete
    final_pert = x_adv - x_query
    final_pert = np.clip(final_pert, -eps, eps)
    
    # ensure perturbed image does not excede original image bounds
    x_adv = np.clip(x_query + final_pert, init_min, init_max)
    
    if vis: return x_adv, perts, pert_grads, map_grads, maps
    return x_adv

# The same version of the attention attack, but it uses np instead of torch to be compatible with tensorflow
def afog_cnn(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255., attn_lr=0.01, vis=False, mode="baseline", meta=None):
    # Select loss function variation based on mode
    attack_modes = {
        "baseline":victim.compute_object_untargeted_gradient,
        "vanishing":victim.compute_object_vanishing_gradient,
        "fabrication":victim.compute_object_fabrication_gradient,
    }
    
    # If visualization is enabled, save intermediate arrays and images
    if vis:
        perts = []
        pert_grads = []
        map_grads = []
        maps = []
    
    # Scale eps according to the range of the image ([0, 1] or [0, 255] typically)
    init_min = min(0.0, np.min(x_query).item())
    init_max = np.max(x_query).item()
    mult_factor = init_max

    eps_iter = eps_iter * mult_factor
    eps = eps * mult_factor
    
    # Initialize perturbation map and attn_map
    pert = np.random.uniform(-eps, eps, size=x_query.shape)
    attn_map = np.ones((x_query.shape))
       
    # Get detections and initialize pert and attn. Sometimes there is additional metadata to pass in
    detections_query = None
    if meta == None:
        detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    else:
        detections_query = victim.detect(x_query, meta=meta)
        detections_query[0]['meta'] = meta # also want to pass in meta here so that the gradient functions can access it
    
    # Make the first adversarial example
    x_adv = x_query + np.multiply(attn_map, pert)

    for i in range(n_iter):
        
        # save for visualization
        if vis:
            perts.append(pert.copy())
            maps.append(attn_map.copy())

        # first get the gradient for x_adv
        grad = attack_modes[mode](x_adv, detections = detections_query, norm=False)
        
        # compute the two partials. Simple multiplication because this is elementwise, not matmul
        pert_grad = np.multiply(grad, attn_map)
        attn_map_grad = np.multiply(grad, pert)
        
        # update perturbation
        signed_pert_grad = np.sign(pert_grad)
        pert -= eps_iter * signed_pert_grad
        pert = np.clip(pert, -eps, eps)
        
        # update attention
        norm_attn_map_grad = (attn_map_grad - np.mean(attn_map_grad)) / np.std(attn_map_grad)
        attn_map -= attn_lr * norm_attn_map_grad
        
        # save for visualization
        if vis:
            pert_grads.append(pert_grad.copy())
            map_grads.append(attn_map_grad.copy())
    
        # make the next iteration of x_adv
        pert = np.clip(np.multiply(attn_map, pert), -eps, eps)
        x_adv = np.clip(x_query + pert, init_min, init_max)
    
    # create final version of attacked image once iteration is complete
    final_pert = x_adv - x_query
    final_pert = np.clip(final_pert, -eps, eps)
    
    # ensure perturbed image does not excede original image bounds
    x_adv = np.clip(x_query + final_pert, init_min, init_max)
    
    if vis: return x_adv, perts, pert_grads, map_grads, maps
    return x_adv


# TOG attacks for comparison. Credit to https://github.com/git-disl/TOG
def tog_untargeted(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255., mode=None, meta=None):
    init_min = min(0, np.min(x_query).item()) 
    init_max = np.max(x_query).item()
    mult_factor = init_max
    eps_iter = eps_iter * mult_factor
    eps = eps * mult_factor
    
    if meta == None:
        detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    else:
        detections_query = victim.detect(x_query, meta=meta)
        detections_query[0]["meta"] = meta
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    
    x_adv = np.clip(x_query + eta, init_min, init_max)
    for _ in range(n_iter):
        grad = victim.compute_object_untargeted_gradient(x_adv, detections=detections_query)
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        x_adv = np.clip(x_query + eta, init_min, init_max)
    return x_adv

# TOG attacks for comparison. Credit to https://github.com/git-disl/TOG
def tog_untargeted_viz(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255.):
    etas = []
    grads = []
    
    detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    etas.append(eta)
    x_adv = np.clip(x_query + eta, 0.0, 1.0)
    for _ in range(n_iter):
        grad = victim.compute_object_untargeted_gradient(x_adv, detections=detections_query)
        grads.append(grad)
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        etas.append(eta)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
    
    return x_adv, etas, grads
