import numpy as np

def collect_states(
    plannable_state,
    state_key_func=lambda s: np.asarray(s.observation()).tobytes()
):
    """
    Returns the list of all states reachable from
    plannable_state's all_init.
    
    Arguments:
        plannable_state: A plannable state that is used to
                         collect all states reachable from all_init.
        key_func: A function that converts a state into a
                  dictionary key so that equal states have
                  equal keys and distinct states have distinct
                  keys (this is necessary because in Python
                  the default behaviour is that equality checks
                  are done by identity and not by value).
    """
    to_expand = [s for s, _ in plannable_state.all_init()]
    states = {}
    
    while len(to_expand):
        s = to_expand.pop()
        states[state_key_func(s)] = s
                
        for a in s.legal_actions():
            for ns, _ in s.all_next(a):
                ns_key = state_key_func(ns)
                if not ns_key in states:
                    to_expand.append(ns)
                    states[ns_key] = ns
    
    return list(states.values())
