import psycopg2


class PostGreEnv:
    def __init__(self):
        self.database_conn = psycopg2.connect(
            database="db_test",
            user="postgres",
            password="12345678",
            host="127.0.0.1",
            port="5432"
        )

    def get_effect_of_this(self, argument, des):
        return 0

    def get_current_argument(self):
        return

    def step(self, action):
        reward = 0
        if action in (1, 2):
            des = True if action == 1 else False
            new_state, reward = self.get_effect_of_this('hy1', des)
        elif action in (3, 4):
            des = True if action == 1 else False
            new_state, reward = self.get_effect_of_this('hy1', des)
        elif action in (3, 4):
            des = True if action == 1 else False
            new_state, reward = self.get_effect_of_this('hy1', des)
        elif action in (3, 4):
            des = True if action == 1 else False
            new_state, reward = self.get_effect_of_this('hy1', des)
        elif action in (3, 4):
            des = True if action == 1 else False
            new_state, reward = self.get_effect_of_this('hy1', des)
        elif action in (3, 4):
            des = True if action == 1 else False
            new_state, reward = self.get_effect_of_this('hy1', des)
        elif action in (3, 4):
            des = True if action == 1 else False
            new_state, reward = self.get_effect_of_this('hy1', des)
        else:
            new_state = self.get_current_argument()
            reward = -0.001
        return new_state, reward, False, {}

    def init_env_argument(self):
        return

    def reset(self):
        self.init_env_argument()
        return
