from .replay_buffer import ReplayBuffer
import itertools

class EndEpisodeSignal(Exception):
    pass

class EndTrainingSignal(Exception):
    pass

class TrainInfo:
    pass

class Trainer:
    def __init__(
        self, algo, policy=None, replay_buffer=None,
        on_begin_episode=None, on_end_episode=None,
        on_begin_step=None, on_end_step=None, verbose=True
    ):
        self.algo = algo
        self.policy = policy
        self.replay_buffer = (replay_buffer if not replay_buffer is None
            else ReplayBuffer(max_size=1, batch_size=1))
        self.on_begin_episode = on_begin_episode or []
        self.on_end_episode = on_end_episode or []
        self.on_begin_step = on_begin_step or []
        self.on_end_step = on_end_step or []
        self.verbose = verbose
        self.reset_stats()

    def reset_stats(self):
        self.train_info = ti = TrainInfo()
        ti.max_steps = None
        ti.max_episodes = None
        ti.max_episode_steps = None
        ti.batch_size = None
        ti.env = None
        ti.step = 0
        ti.episode = 0
        ti.episode_step = 0
        ti.done = True
        ti.obs = None
        ti.transition = (None, None, None, None)
        ti.interrupted = False
        ti.terminated = False
        ti.starting = None

    def train(self, env, max_steps=None, max_episodes=None,
        max_episode_steps=None, batch_size=None, actions=None
    ):
        if max_steps is None and max_episodes is None:
            raise ValueError("At least one of max_steps and max_episodes must be specified.")

        ti = self.train_info
        ti.max_steps = max_steps
        ti.max_episodes = max_episodes
        ti.max_episode_steps = max_episode_steps
        ti.batch_size = batch_size
        ti.env = env
        ti.done = True
        ti.starting = True
        ti.interrupted = False
        ti.terminated = False
        ti.episode_step = 0

        if actions is None:
            if self.policy is None:
                raise ValueError("Either a policy or an ``actions`` sequence needs to be specified.")
        else:
            actions = iter(itertools.cycle(actions))

        while True:
            try:
                # new episode
                if ti.done or ti.interrupted or ti.terminated or (
                    not ti.max_episode_steps is None and ti.episode_step >= ti.max_episode_steps
                ):
                    if ti.starting:
                        ti.starting = False
                    else:
                        if self.verbose:
                            print(f"Episode {ti.episode} finished after {ti.episode_step} steps. Taken the total of {ti.step} steps.")

                        for callback in self.on_end_episode:
                            callback(ti.step, ti)

                    if (
                        ti.terminated or
                        (not ti.max_episodes is None and ti.episode >= ti.max_episodes) or 
                        (not ti.max_steps is None and ti.step >= ti.max_steps)
                    ):
                        break

                    ti.obs = ti.env.reset()
                    ti.done = False
                    ti.episode_step = 0
                    ti.interrupted = False

                    for callback in self.on_begin_episode:
                        callback(ti.step, ti)

                    ti.episode += 1

                if not ti.max_steps is None and ti.step >= ti.max_steps:
                    ti.interrupted = True
                    continue

                for callback in self.on_begin_step:
                    callback(ti.step, ti)

                if actions is None:
                    a = self.policy(ti.obs)
                else:
                    a = next(actions)

                obs_next, reward, ti.done, info = ti.env.step(a)
                ti.transition = (ti.obs, a, reward, obs_next, ti.done, info)
                self.replay_buffer.add(*ti.transition)
                batch = self.replay_buffer.sample(batch_size=ti.batch_size)
                self.algo(*batch)

                for callback in self.on_end_step:
                    callback(ti.step, ti)
                
                ti.obs = obs_next
                ti.episode_step += 1
                ti.step += 1
            except EndEpisodeSignal:
                ti.interrupted = True
            except EndTrainingSignal:
                ti.terminated = True
