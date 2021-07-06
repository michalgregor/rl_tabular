import numpy as np

def value_iteration(
    vtable, states,
    num_episodes=50, discount=0.9,
    verbose=True
):
    for episode in range(num_episodes):
        if verbose:
            print("Iteration {} started.".format(episode))

        for state in states:
            if state.is_done():
                continue

            legals = state.legal_actions()
            maxval = -np.inf
            
            for a in legals:
                val = 0
                
                for next_state, prob in state.all_next(a):
                    r = next_state.rewards()
                    val += prob * (r + discount*vtable[next_state])
                    
                maxval = max(val, maxval)
                        
            vtable[state] = maxval if len(legals) else 0
