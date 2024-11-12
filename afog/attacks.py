from attack_utils.target_utils import generate_attack_targets
import numpy as np
import torch

def afog(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255., attn_lr=0.01, vis=False, mode="baseline", meta=None):
    attack_modes = {
        "baseline":victim.compute_object_untargeted_gradient,
        "vanishing":victim.compute_object_vanishing_gradient,
        "fabrication":victim.compute_object_fabrication_gradient,
    }
    
    if vis:
        etas = []
        eta_grads = []
        map_grads = []
        maps = []
    
    init_min = min(0.0, torch.min(x_query).item())
    init_max = torch.max(x_query).item()
    mult_factor = init_max

    eps_iter = eps_iter * mult_factor
    eps = eps * mult_factor
    
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    attn_map = np.ones((x_query.shape))
       
    # Get detections and initialize eta and attn. Sometimes there is additional metadata to pass in
    detections_query = None
    if meta == None:
        detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    else:
        detections_query = victim.detect(x_query, meta=meta)
        detections_query[0]['meta'] = meta # also want to pass in meta here so that the gradient functions can access it
    
    # Make the first adversarial example
    x_adv = x_query + np.multiply(attn_map, eta)
    assert torch.isfinite(x_adv).all(), "x_adv on initialization contains infinite or NaN!"

    #x_adv = x_query + np.multiply(attn_map, eta)
    for i in range(n_iter):
        
        # save for visualization
        if vis:
            etas.append(eta.copy())
            maps.append(attn_map.copy())

        # handle NaN cases, which seem to happen very rarely
        x_adv = torch.nan_to_num(x_adv, nan=0.0, posinf=10000, neginf=-10000)

        # first get the gradient for x_adv
        grad = attack_modes[mode](x_adv, x_query, detections = detections_query, norm=False)
        assert np.isfinite(grad).all(), "grad contains infinite or NaN!"
        
        # compute the two partials. Simple multiplication because this is elementwise, not matmul
        eta_grad = np.multiply(grad, attn_map)
        attn_map_grad = np.multiply(grad, eta)# + np.random.normal(0, 0.01, (x_query.shape))
        
        # update eta
        signed_eta_grad = np.sign(eta_grad)
        eta -= eps_iter * signed_eta_grad
        eta = np.clip(eta, -eps, eps)
        
        # update attention
        norm_attn_map_grad = (attn_map_grad - np.mean(attn_map_grad)) / np.std(attn_map_grad)
        attn_map -= attn_lr * norm_attn_map_grad
        
        # save for visualization
        if vis:
            eta_grads.append(eta_grad.copy())
            map_grads.append(attn_map_grad.copy())
    
        # make the next iteration of x_adv
        #x_adv = x_query + np.multiply(attn_map, eta)
        pert = np.clip(np.multiply(attn_map, eta), -eps, eps)
        x_adv = np.clip(x_query + pert, init_min, init_max)
        assert np.isfinite(x_adv).all(), "x_adv after clipping infinite or NaN!"
    
    final_pert = x_adv - x_query
    final_pert = np.clip(final_pert, -eps, eps)
    x_adv = np.clip(x_query + final_pert, init_min, init_max)
    assert torch.isfinite(x_adv).all(), "final x_adv contains infinite or NaN!"
    
    if vis: return x_adv, etas, eta_grads, map_grads, maps
    return x_adv

# The same version of the attention attack, but it uses np instead of torch to be compatible with tensorflow
def afog_cnn(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255., attn_lr=0.01, vis=False, mode="baseline", meta=None):
    attack_modes = {
        "baseline":victim.compute_object_untargeted_gradient,
        "vanishing":victim.compute_object_vanishing_gradient,
        "fabrication":victim.compute_object_fabrication_gradient
    }
    
    if vis:
        etas = []        
        eta_grads = []
        map_grads = []
        maps = []
   
    # Get detections and initialize eta and attn. Sometimes there is additional metadata to pass in
    detections_query = None
    if meta == None:
        detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    else:
        detections_query = victim.detect(x_query, meta=meta)
        detections_query[0]['meta'] = meta # also want to pass in meta here so that the gradient functions can access it
            
    init_min = min(0.0, np.min(x_query).item())
    init_max = np.max(x_query).item()
    mult_factor = init_max

    eps_iter = eps_iter * mult_factor
    eps = eps * mult_factor
    
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    attn_map = np.ones((x_query.shape))
    
    # initialize attention maps with bounding boxes. Area inside the box gets a 1.5 effect multiplier
#     for det in detections_query:
#         effect = np.ones(attn_map.shape)
#         xmin, ymin, xmax, ymax = int(det[-4]), int(det[-3]), int(det[-2]), int(det[-1])
#         effect[0, ymin:ymax, xmin:xmax, :] = 1.5
#         attn_map = np.multiply(attn_map, effect)
    
    # Make the first adversarial example
    x_adv = x_query + np.multiply(attn_map, eta)
    assert np.isfinite(x_adv).all(), "x_adv on initialization contains infinite or NaN!"

    #x_adv = x_query + np.multiply(attn_map, eta)
    for i in range(n_iter):
        
        # save for visualization
        if vis:
            etas.append(eta.copy())
            maps.append(attn_map.copy())

        # handle NaN cases, which seem to happen very rarely
        x_adv = np.nan_to_num(x_adv, nan=0.0, posinf=10000, neginf=-10000)

        # first get the gradient for x_adv
        grad = attack_modes[mode](x_adv, detections = detections_query, norm=False)
        assert np.isfinite(grad).all(), "grad contains infinite or NaN!"
        
        # compute the two partials. Simple multiplication because this is elementwise, not matmul
        eta_grad = np.multiply(grad, attn_map)
        attn_map_grad = np.multiply(grad, eta)# + np.random.normal(0, 0.01, (x_query.shape))
        
        # update eta
        signed_eta_grad = np.sign(eta_grad)
        eta -= eps_iter * signed_eta_grad
        eta = np.clip(eta, -eps, eps)
        
        # update attention
        norm_attn_map_grad = (attn_map_grad - np.mean(attn_map_grad)) / (0.001 + np.std(attn_map_grad))
        attn_map -= attn_lr * norm_attn_map_grad
        
        # save for visualization
        if vis:
            eta_grads.append(eta_grad.copy())
            map_grads.append(attn_map_grad.copy())
    
        # make the next iteration of x_adv
        #x_adv = x_query + np.multiply(attn_map, eta)
        pert = np.clip(np.multiply(attn_map, eta), -eps, eps)
        x_adv = np.clip(x_query + pert, init_min, init_max)
        assert np.isfinite(x_adv).all(), "x_adv after clipping infinite or NaN!"
    
    final_pert = x_adv - x_query
    final_pert = np.clip(final_pert, -eps, eps)
    x_adv = np.clip(x_query + final_pert, init_min, init_max)
    assert np.isfinite(x_adv).all(), "final x_adv contains infinite or NaN!"
    
    if vis: return x_adv, etas, eta_grads, map_grads, maps
    return x_adv

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
