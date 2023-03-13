from .value_functions import compute_action_value
import numpy as np

def action_value_control(env, action_val_func, max_steps=None):
    env.reset()

    step = 0
    done = False

    while not done and (max_steps is None or step < max_steps):
        maxa = None
        maxval = -np.inf
        state = env.single_plannable_state()

        for a in state.legal_actions():
            val = action_val_func(state, a)
            if val > maxval:
                maxval = val
                maxa = a

        obs, reward, terminated, truncated, info = env.step(maxa)
        done = terminated or truncated

        step += 1

def qtable_control(env, qtable, **kwargs):
    return action_value_control(env, lambda state, a: qtable[state, a], **kwargs)

def vtable_control(env, vtable, discount=0.9, **kwargs):
    def action_val_func(state, a):
        return compute_action_value(state, a, vtable, discount=discount)
    return action_value_control(env, action_val_func, **kwargs)
