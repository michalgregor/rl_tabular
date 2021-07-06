import numpy as np
import matplotlib.pyplot as plt
from gym_plannable.env.grid_world import Actor, MazeEnv
from .states import collect_states

def get_state_value_array(vtable, observation_space,
                          states, default=0):
    low, high = observation_space.low, observation_space.high
    span = high - low
    assert len(span) == 2
    value_array = np.zeros(span)
    
    for state in states:
        obs = state.observations() - low
        value_array[obs[0], obs[1]] = vtable.get(state, default)
        
    return value_array

def get_action_value_array(
    qtable, observation_space, states,
    actions=[0, 1, 2, 3], default=0
):
    low, high = observation_space.low, observation_space.high
    span = high - low
    assert len(span) == 2
    value_array = np.zeros(tuple(span) + (len(actions),))
    
    for state in states:
        for action in actions:
            obs = state.observations() - low
            value_array[obs[0], obs[1], action] = qtable.get(state, action, default)
        
    return value_array

def plot_state_values(vtable, states=None, env=None,
                      default=0, ax=None, cmap='OrRd', render_env=True,
                      render_agent=False, cbar=True, update_display=False,
                      update_delay=None, **kwargs):
    interrupt = False
    try:
        if env is None:
            env = MazeEnv()
            env.reset()

        if states is None:
            states = collect_states(env.single_plannable_state())

        value_array = get_state_value_array(
            vtable, env.observation_space, states, default=default)
        low, high = env.observation_space.low, env.observation_space.high
        
        if ax is None and update_display:
            fig = env.render_fig
            if not fig is None:
                fig.clf()
                ax = fig.gca()

        if ax is None:
            ax = plt.gca()
            
        fig = ax.get_figure()
            
        if render_env:
            if render_agent:
                render_sequence = None
            else:
                state = env.single_plannable_state()
                render_sequence = [obj for obj in state.render_sequence if not isinstance(obj, Actor)]
            
            env.render(fig=fig, ax=ax, render_sequence=render_sequence)
        
        m = ax.matshow(value_array,
            extent=(low[0]-0.5, high[0]-0.5, high[1]-0.5, low[1]-0.5),
            cmap=cmap, **kwargs
        )
        
        if cbar:
            vmin = value_array.min()
            vmax = value_array.max()
            
            if vmin == vmax:
                m.norm.vmin = vmin
                m.norm.vmax = vmin+1
            
            fig.colorbar(ax=ax, mappable=m)
        
        if update_display:
            env.update_display(delay=update_delay)

        return m
    
    except KeyboardInterrupt:
        interrupt = True

    if interrupt:
        return KeyboardInterrupt()

def plot_action_values(
    qtable, states, env=None,
    actions=[0, 1, 2, 3], default=0, ax=None,
    cmap='OrRd', render_agent=False, render_env=True,
    temperature=10, update_display=False,
    update_delay=None, **kwargs
):
    interrupt = False
    try:
        actionDirections = [
            (-1, 0), (1, 0), (0, -1), (0, 1)
        ]

        if env is None:
            env = MazeEnv()
            env.reset()

        if states is None:
            states = collect_states(env.single_plannable_state())
        
        value_array = get_action_value_array(
            qtable, env.observation_space, states,
            default=default, actions=actions)
        
        if ax is None and update_display:
            fig = env.render_fig
            if not fig is None:
                ax = fig.axes[0] if len(fig.axes) else fig.gca()

        if ax is None:
            ax = plt.gca()
            
        cmap = plt.get_cmap(cmap)
        fig = ax.get_figure()
            
        if render_env:
            if render_agent:
                render_sequence = None
            else:
                state = env.single_plannable_state()
                render_sequence = [obj for obj in state.render_sequence if not isinstance(obj, Actor)]
        
            env.render(fig=fig, ax=ax, render_sequence=render_sequence)       
        
        for posX in range(value_array.shape[0]):
            for posY in range(value_array.shape[1]):
                vals = value_array[posX, posY]
                if (vals == 0).all(): continue
                
                exp_vals = np.exp(vals / temperature)
                probs = exp_vals / exp_vals.sum()
                scaled_probs = probs / np.max(probs)
                                
                for ia, p in enumerate(scaled_probs):
                    arrow_size = p
                    arrow_length = 0.25 * arrow_size
                    inc = actionDirections[ia]
                    
                    if arrow_size != 0:
                        ax.arrow(posY, posX,
                            inc[1] * arrow_length, inc[0] * arrow_length,
                            head_width=0.15 * arrow_size, 
                            head_length=0.15 * arrow_size,
                            fc=cmap(arrow_size), ec=cmap(arrow_size),
                            **kwargs
                        )
        
        if update_display:
            env.update_display(delay=update_delay)
    
    except KeyboardInterrupt:
        interrupt = True

    if interrupt:
        raise KeyboardInterrupt()
    