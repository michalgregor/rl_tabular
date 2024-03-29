import numpy as np
import matplotlib.pyplot as plt
from gym_plannable.env.grid_world import Actor, MazeEnv
from .states import collect_states
from .value_functions import StateValueTable, ActionValueTable

import matplotlib.pyplot as plt
import numpy as np
            
def get_state_value_array(vtable, observation_space, states):
    low, high = observation_space.low, observation_space.high
    span = high - low
    assert len(span) == 2
    value_array = np.zeros(span)
    
    for state in states:
        obs = state.observation() - low
        value_array[obs[0], obs[1]] = vtable[state]
        
    return value_array

def get_action_value_array(
    qtable, observation_space, states, num_actions
):
    low, high = observation_space.low, observation_space.high
    span = high - low
    assert len(span) == 2
    value_array = np.zeros(tuple(span) + (num_actions,))
    
    for state in states:
        for action in range(num_actions):
            obs = state.observation() - low
            value_array[obs[0], obs[1], action] = qtable[state, action]
    return value_array

def plot_env(env=None, ax=None, fig=None, render_agent=False, **render_kwargs):
    if env is None:
        env = MazeEnv()
        env.reset()

    if ax is None:
        ax = plt.gca()
        fig = ax.get_figure()

    if render_agent:
        render_sequence = None
    else:
        state = env.single_plannable_state()
        render_sequence = [obj for obj in state.render_sequence if not isinstance(obj, Actor)]
    
    render_kwargs_tmp = dict(fig=fig, ax=ax, render_sequence=render_sequence)
    render_kwargs_tmp.update(render_kwargs)

    env.render(**render_kwargs_tmp)

def plot_path(
    path, tile_size=1,
    show_arrows=True, arrow_color='black', arrow_alpha=1.0,
    show_visited=False, visited_color='blue', visited_alpha=0.1,
    ax=None, env=None, render_env=True, render_agent=False,
    render_kwargs=None, arrow_kwargs=None, visited_kwargs=None,
):
    render_kwargs = dict(dict(
        env=env, ax=ax, render_agent=render_agent
    ), **(render_kwargs or {}))

    arrow_kwargs = dict(dict(
        zorder=5,
        scale_units='xy',
        scale=tile_size,
        color=arrow_color,
        alpha=arrow_alpha,
    ), **(arrow_kwargs or dict()))

    visited_kwargs = dict(dict(
        facecolor=visited_color,
        alpha=visited_alpha
    ), **(visited_kwargs or dict()))

    if ax is None:
        ax = plt.gca()

    if render_env:
        plot_env(**render_kwargs)

    if show_arrows:
        path = np.asarray(path)

        X = path[:-1, 1]
        Y = path[:-1, 0]
        U = path[1:, 1] - path[:-1, 1]
        V = path[:-1, 0] - path[1:, 0]

        ax.quiver(X, Y, U, V, **arrow_kwargs)

    if show_visited:
        for r, c in path:
            patch = plt.Rectangle(
                [c - tile_size / 2,
                r - tile_size / 2],
                tile_size, tile_size,
                **visited_kwargs
            )
            ax.add_patch(patch)

def plot_state_values(vtable, states=None, env=None,
                      ax=None, cmap='OrRd', render_env=True,
                      render_agent=False, cbar=True, cax=None,
                      update_display=False,
                      update_delay=None, render_kwargs=None, **kwargs):
    interrupt = False
    render_kwargs = render_kwargs or dict()

    try:
        if env is None:
            env = MazeEnv()
            env.reset()

        if states is None:
            states = collect_states(env.single_plannable_state())

        value_array = get_state_value_array(
            vtable, env.observation_space, states)
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
            plot_env(env=env, ax=ax, fig=fig, render_agent=render_agent, **render_kwargs)
        
        m = ax.matshow(value_array,
            extent=(low[1]-0.5, high[1]-0.5, high[0]-0.5, low[0]-0.5),
            cmap=cmap, **kwargs
        )
        
        if cbar:
            vmin = value_array.min()
            vmax = value_array.max()
            
            if vmin == vmax:
                m.norm.vmin = vmin
                m.norm.vmax = vmin+1
            
            if cax is None:
                fig.colorbar(ax=ax, mappable=m)
            else:
                fig.colorbar(cax=cax, mappable=m)
        
        if update_display:
            env.update_display(delay=update_delay)

        return m
    
    except KeyboardInterrupt:
        interrupt = True

    if interrupt:
        raise KeyboardInterrupt()

def plot_action_values(
    qtable, states, env=None,
    action_spec=None, ax=None,
    cmap='OrRd', render_agent=False, render_env=True,
    temperature=10, update_display=False,
    update_delay=None, render_kwargs=None, **kwargs
):
    interrupt = False
    render_kwargs = render_kwargs or dict()

    try:
        action_spec = action_spec or env.action_spec
        
        if env is None:
            env = MazeEnv()
            env.reset()

        if states is None:
            states = collect_states(env.single_plannable_state())
        
        value_array = get_action_value_array(
            qtable, env.observation_space, states,
            num_actions=len(action_spec))
        
        if ax is None and update_display:
            fig = env.render_fig
            if not fig is None:
                ax = fig.axes[0] if len(fig.axes) else fig.gca()

        if ax is None:
            ax = plt.gca()
            
        cmap = plt.get_cmap(cmap)
        fig = ax.get_figure()
            
        if render_env:
            plot_env(env=env, ax=ax, render_agent=render_agent, fig=fig, **render_kwargs)  
        
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
                    inc = action_spec[ia]
                    
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

class Plotter:
    def __init__(self, env, *args, figsize=None, update_display=True,**kwargs):
        self.env = env
        self.tab_kwargs = []
        self.tab_func = []
        self.tab_axes = []
        self.states = collect_states(env.single_plannable_state())

        if not figsize is None:
            self.env.render_fig.set_size_inches(figsize)
        
        ax_spec = []
        for tab_type in args:
            if isinstance(tab_type, tuple):
                tab_type, _ = tab_type

            if tab_type == StateValueTable:
                ax_spec.extend((1, 0.025)) # for ax and cax
            elif tab_type == ActionValueTable:
                ax_spec.append(1) # for ax
            else:
                raise TypeError(f"Unknown type of table '{tab_type}'.")

        axes = self.env.render_fig.subplots(
            1, len(ax_spec), gridspec_kw={'width_ratios': ax_spec}
        )

        if len(ax_spec) == 1: axes = [axes]
        iax = 0

        for tab_type in args:
            if isinstance(tab_type, tuple):
                tab_type, inst_kwargs = tab_type
            else:
                inst_kwargs = {}

            tab_kwargs = {'env': self.env, 'render_agent': True,
                          'update_display': False, 'states': self.states}

            if tab_type == StateValueTable:
                self.tab_func.append(plot_state_values)
                self.tab_axes.append((axes[iax], axes[iax+1]))
                tab_kwargs.update({'ax': axes[iax], 'cax': axes[iax+1]})
                iax += 2
            elif tab_type == ActionValueTable:
                self.tab_func.append(plot_action_values)
                self.tab_axes.append((axes[iax],))
                tab_kwargs.update({'ax': axes[iax]})
                iax += 1
            else:
                raise TypeError(f"Unexpected type '{tab_type}'.")

            tab_kwargs.update(kwargs)
            tab_kwargs.update(inst_kwargs)
            self.tab_kwargs.append(tab_kwargs)

        self.tab_kwargs[-1].update({'update_display': update_display})

    def plot(self, *args, **kwargs):    
        for tab, func, tab_kwargs, axes in zip(
            args, self.tab_func, self.tab_kwargs, self.tab_axes
        ):
            if isinstance(tab, tuple):
                tab, inst_kwargs = tab
            else:
                inst_kwargs = {}

            tab_kwargs = tab_kwargs.copy()
            tab_kwargs.update(kwargs)
            tab_kwargs.update(inst_kwargs)

            for ax in axes: ax.clear()
            func(tab, **tab_kwargs)
