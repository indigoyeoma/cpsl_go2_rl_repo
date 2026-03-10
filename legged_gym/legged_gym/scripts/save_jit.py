import os, sys
from statistics import mode
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic import Actor, StateHistoryEncoder, get_activation, ActorCriticRMA
from rsl_rl.modules.estimator import Estimator
from rsl_rl.modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone, DepthOnlyFCBackbone128x96
import argparse
import code
import shutil
from legged_gym import LEGGED_GYM_ROOT_DIR

def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="model"):
    if not os.path.isdir(root):  # use first 4 chars to mactch the run name
        model_name_cand = os.path.basename(root)
        model_parent = os.path.dirname(root)
        model_names = os.listdir(model_parent)
        model_names = [name for name in model_names if os.path.isdir(os.path.join(model_parent, name))]
        for name in model_names:
            if len(name) >= 6:
                if name[:6] == model_name_cand:
                    root = os.path.join(model_parent, name)
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(root, model)
    return load_path, checkpoint

class HardwareVisionNN(nn.Module):
    def __init__(self,  num_prop,
                        num_scan,
                        num_priv_latent, 
                        num_priv_explicit,
                        num_hist,
                        num_actions,
                        tanh,
                        actor_hidden_dims=[512, 256, 128],
                        scan_encoder_dims=[128, 64, 32],
                        depth_encoder_hidden_dim=512,
                        activation='elu',
                        priv_encoder_dims=[64, 20]
                        ):
        super(HardwareVisionNN, self).__init__()

        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.num_priv_explicit = num_priv_explicit
        num_obs = num_prop + num_scan + num_hist*num_prop + num_priv_latent + num_priv_explicit
        self.num_obs = num_obs
        activation = get_activation(activation)
        
        # Check standard depths
        if num_scan == 32: # Output dim
            # Check implicit image size from encoder dims if possible, or just assume standard based on usage
            pass
            
        # For this script, we want to force the specific backbone if we know the resolution
        # But HardwareVisionNN is generic. Let's look at how it initializes actor.
        
        # We need to ensure the Actor uses the right backbone. 
        # rsl_rl Actor usually takes a backbone string or class.
        # But here it seems to instantiate it internally or we pass it?
        
        # Actually, looking at rsl_rl Actor (not visible here but inferred), it likely uses the config.
        # However, for this export script, we are loading state dicts.
        # We need to construct a model structure that MATCHES the trained model.
        
        # If the trained model was 128x96, we need to make sure the backbone here matches.
        # The 'scan_encoder_dims' arg is [128, 64, 32] which matches our 128x96 backbone output structure?
        # No, the backbone has: Conv(32) -> MaxPool -> Conv(64) -> Flatten -> Linear(128) -> Linear(32)
        
        # Wait, HardwareVisionNN initializes Actor. 
        # Standard Actor in rsl_rl (from checking previous files or assuming standard)
        # usually builds the backbone based on config.
        
        # FOR THIS EXPORT SCRIPT:
        # We need to instantiate the specific 128x96 backbone and assign it, 
        # OR ensure Actor builds it.
        
        # Since we can't easily change Actor.__init__ signatures from here easily without seeing them,
        # let's look at how we can 'hotswap' the backbone or if we should just make HardwareVisionNN use it explicitly.
        
        self.actor = Actor(num_prop, num_scan, num_actions, scan_encoder_dims, actor_hidden_dims, priv_encoder_dims, num_priv_latent, num_priv_explicit, num_hist, activation, tanh_encoder_output=tanh)

        self.estimator = Estimator(input_dim=num_prop, output_dim=num_priv_explicit, hidden_dims=[128, 64])
        
    def forward(self, obs, depth_latent):
        obs[:, self.num_prop+self.num_scan : self.num_prop+self.num_scan+self.num_priv_explicit] = self.estimator(obs[:, :self.num_prop])
        return self.actor(obs, hist_encoding=True, eval=False, scandots_latent=depth_latent)
        # return obs, depth_latent

def play(args):
    load_run = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', args.proj_name, args.exptid)
    checkpoint = args.checkpoint

    n_priv_explicit = 3 + 3 + 3
    n_priv_latent = 4 + 1 + 12 +12
    num_scan = 132
    num_actions = 12
    
    # depth_buffer_len = 2
    depth_resized = (128, 96)  # Matches training go2_student_config.py resized = (128, 96)
    
    n_proprio = 3 + 2 + 3 + 4 + 36 + 4 +1
    history_len = 10

    device = torch.device('cpu')
    policy = HardwareVisionNN(n_proprio, num_scan, n_priv_latent, n_priv_explicit, history_len, num_actions, args.tanh, scan_encoder_dims=[128]).to(device)
    load_path, checkpoint = get_load_path(root=load_run, checkpoint=checkpoint)
    load_run = os.path.dirname(load_path)
    print(f"Loading model from: {load_path}")
    ac_state_dict = torch.load(load_path, map_location=device)
    # policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    
    # HACK: Manually swap the backbone in the actor if we want to trace the correct one
    # The loaded state dict will have shapes for 128x96 if that's what was trained.
    # But self.actor defaults might be creating the wrong backbone.
    # We should explicitly set it.
    
    policy.actor.scan_encoder = DepthOnlyFCBackbone128x96(
        prop_dim=n_proprio, 
        scandots_output_dim=128, 
        hidden_state_dim=512, 
        output_activation=None
    )
    
    policy.actor.load_state_dict(ac_state_dict['depth_actor_state_dict'], strict=False)
    
    # Strip 'base_backbone.' prefix if present inside the depth encoder state dict
    depth_encoder_state_dict = ac_state_dict['depth_encoder_state_dict']
    new_depth_encoder_state_dict = {}
    for k, v in depth_encoder_state_dict.items():
        if k.startswith('base_backbone.'):
            new_depth_encoder_state_dict[k.replace('base_backbone.', '')] = v
        else:
            new_depth_encoder_state_dict[k] = v
    
    policy.actor.scan_encoder.load_state_dict(new_depth_encoder_state_dict, strict=False)
    policy.estimator.load_state_dict(ac_state_dict['estimator_state_dict'])
    
    policy = policy.to(device)#.cpu()
    if not os.path.exists(os.path.join(load_run, "traced")):
        os.mkdir(os.path.join(load_run, "traced"))
        
    torch.save(ac_state_dict['depth_encoder_state_dict'], os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-vision_weight.pt"))

    # Save the traced actor
    policy.eval()
    with torch.no_grad(): 
        num_envs = 1
        
        obs_input = torch.ones(num_envs, n_proprio + num_scan + n_priv_explicit + n_priv_latent + history_len*n_proprio, device=device)
        depth_latent = torch.ones(1, 128, device=device)
        test = policy(obs_input, depth_latent)
        
        traced_policy = torch.jit.trace(policy, (obs_input, depth_latent))
        
        # traced_policy = torch.jit.script(policy)
        save_path = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-base_jit.pt")
        traced_policy.save(save_path)
        print("Saved traced_actor at ", os.path.abspath(save_path))
        
        # Copy to deployment folder
        # Relative path from LEGGED_GYM_ROOT_DIR (legged_gym/legged_gym)
        # ../../../ goes to ws_go2
        deploy_policy_dir = os.path.abspath(os.path.join(LEGGED_GYM_ROOT_DIR, "../../../cpsl_unitree_sdk2_python/deploy_go2_lowlevel/policy/"))
        if os.path.exists(deploy_policy_dir):
            vision_src = os.path.join(load_run, "traced", args.exptid + "-" + str(checkpoint) + "-vision_weight.pt")
            vision_dst = os.path.join(deploy_policy_dir, args.exptid + "-" + str(checkpoint) + "-vision_weight.pt")
            shutil.copy(vision_src, vision_dst)
            
            base_src = save_path
            base_dst = os.path.join(deploy_policy_dir, args.exptid + "-" + str(checkpoint) + "-base_jit.pt")
            shutil.copy(base_src, base_dst)
            
            print(f"Copied JIT files to deployment folder: {deploy_policy_dir}")
        else:
            print(f"Warning: Deployment policy directory not found: {deploy_policy_dir}")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_name', type=str, default='parkour_new', help='Project name (go2_student, go2_teacher, etc.)')
    parser.add_argument('--exptid', type=str)
    parser.add_argument('--checkpoint', type=int, default=-1)
    parser.add_argument('--tanh', action='store_true')
    args = parser.parse_args()
    play(args)
    