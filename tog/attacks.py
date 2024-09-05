from attack_utils.target_utils import generate_attack_targets
import numpy as np


def tog_attention(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255., attn_lr=0.01):
    detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    attn_map = np.random.normal(1, 0.05, (x_query.shape))
    for i in range(n_iter):
        # on odd iterations, update eta
        if i % 2 == 0: 
            grad = victim.compute_object_attention_gradient(x_query, eta, attn_map, detections_query, mode="eta")
            signed_grad = np.sign(grad)
            eta += eps_iter * signed_grad
        
        # on even iterations, update attention map
        else:
            grad = victim.compute_object_attention_gradient(x_query, eta, attn_map, detections_query, mode="attn")
            attn_map += attn_lr * grad
    eta = np.clip(eta, -eps, eps)
    x_adv = x_query + np.multiply(eta, attn_map)
    x_adv = np.clip(x_adv, 0.0, 1.0)
    return x_adv
            
                      

def tog_vanishing(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255.):
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    x_adv = np.clip(x_query + eta, 0.0, 1.0)
    for _ in range(n_iter):
        grad = victim.compute_object_vanishing_gradient(x_adv)
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
    return x_adv


def tog_fabrication(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255.):
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    x_adv = np.clip(x_query + eta, 0.0, 1.0)
    for _ in range(n_iter):
        grad = victim.compute_object_fabrication_gradient(x_adv)
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
    return x_adv


def tog_mislabeling(victim, x_query, target, n_iter=10, eps=8/255., eps_iter=2/255.):
    detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    detections_target = generate_attack_targets(detections_query, confidence_threshold=victim.confidence_thresh_default,
                                                mode=target)
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    x_adv = np.clip(x_query + eta, 0.0, 1.0)
    for _ in range(n_iter):
        grad = victim.compute_object_mislabeling_gradient(x_adv, detections=detections_target)
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
    return x_adv


def tog_untargeted(victim, x_query, n_iter=10, eps=8/255., eps_iter=2/255.):
    detections_query = victim.detect(x_query, conf_threshold=victim.confidence_thresh_default)
    eta = np.random.uniform(-eps, eps, size=x_query.shape)
    x_adv = np.clip(x_query + eta, 0.0, 1.0)
    for _ in range(n_iter):
        grad = victim.compute_object_untargeted_gradient(x_adv, detections=detections_query)
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
    return x_adv
