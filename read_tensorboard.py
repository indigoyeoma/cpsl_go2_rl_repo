import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Load the most recent log
log_path = '/home/jiwoo/ws/go2_rl/logs/rough_go2/Aug26_22-12-33_'
ea = EventAccumulator(log_path)
ea.Reload()

print('Available scalar tags:')
tags = ea.Tags()['scalars']
for tag in sorted(tags)[:15]:  # Show first 15 sorted
    print('  ' + tag)

print('\n--- Key Metrics (Latest Values) ---')
# Get the latest values for key metrics
for metric_name in ['Train/mean_reward', 'Episode/rew_lin_vel_xy', 'Episode/rew_collision', 'Episode/rew_base_height', 'Episode/rew_orientation']:
    if metric_name in tags:
        data = ea.Scalars(metric_name)
        if data:
            print('{}: {:.4f} (step {})'.format(metric_name, data[-1].value, data[-1].step))