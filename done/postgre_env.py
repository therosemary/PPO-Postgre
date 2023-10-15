import time
from io import StringIO

import psycopg2
import torch
from faker import Faker
from interval3 import Interval

from sql_str import *

max_memory = 1024 * 1024 * 8
min_maintenance_work_mem = 1

argument_name = [
                 # 'shared_buffer',
                 # 'wal_buffers',  # WAL缓冲区大小
                 'maintenance_work_mem',  # 维护任务的mem
                 'checkpoint_timeout',  # 数据持久化策略延迟
                 'bgwriter_delay',  # 刷shared buffer脏页的进程调度间隔
                 'bgwriter_lru_maxpages',  # 一次最多刷多少脏页
                 'bgwriter_lru_multiplier',  # 一次扫描多少个块 = 脏页数倍数
                 'work_mem'  # 增强排序能力 但是不适宜过大
]
default_argument = [
    65536,
    300,
    200,
    100,
    2,
    4096]

argument_name_dict = {
    'maintenance_work_mem': 0,
    'checkpoint_timeout': 1,
    'bgwriter_delay': 2,
    'bgwriter_lru_maxpages': 3,
    'bgwriter_lru_multiplier': 4,
    'work_mem': 5,
}

argument_range = [
    [1024, max_memory / 4],
    [30, 3600],
    [10, 10000],
    [1, 1000],
    [1, 10],
    [64, 524288]
]


class Shape:
    def __init__(self, shape_, n):
        self.shape = shape_
        self.n = n


class PostGreEnv:
    def __init__(self):
        self.database_conn = psycopg2.connect(
            database="ppo2",
            user="postgres",
            password="8105",
            host="127.0.0.1",
            port="5432"
        )
        self.state = torch.Tensor(default_argument)
        self.observation_space = Shape((6, ), 1)
        self.action_space = Shape((), 12)
        self.first_query_cost = 0
        self.query_cost_now = 0

    @staticmethod
    def query_test(database_conn):
        s = time.time()
        cur = database_conn.cursor()
        cur.execute(query_sql)
        # res = cur.fetchall()
        cost = time.time() - s
        return cost

    def get_effect_of_this(self, argument, des):
        # 修改参数并测试给出奖励
        reward = 0
        database_conn = self.database_conn
        cur = database_conn.cursor()
        new_state, is_change = self.check_argument(argument_name_dict[argument], des)
        self.state = torch.Tensor(new_state)
        if is_change:
            old_isolation_level = database_conn.isolation_level
            database_conn.set_isolation_level(0)
            time_before = self.query_test(database_conn)
            alter_sql = self.get_alter_sql(argument, new_state[argument_name_dict[argument]])
            reload_sql = 'select pg_reload_conf();'
            cur.execute(alter_sql)
            cur.execute(reload_sql)

            database_conn.set_isolation_level(old_isolation_level)
            cur.execute('show shared_buffers;')
            time_after = self.query_test(database_conn)
            time_change = time_after - time_before
            self.query_cost_now = time_after
            print('参数： {}, 动作: {}'.format(argument, '降低' if des else '升高'))
            print('本次查询相比上次查询多了： {}s'.format(time_change))
            if time_change > 0:
                reward = -time_change
            else:
                reward = time_change

        return new_state, reward

    @staticmethod
    def get_alter_sql(argument, value):
        return """
            ALTER SYSTEM SET {}={};
        """.format(argument, value)

    def check_argument(self, index, des):
        before = self.state.view(-1).numpy().tolist()
        new_state = before
        if des:
            new_ = int(new_state[index] * 0.9)
            if new_ == new_state[index]:
                new_state[index] = new_ - 1
            else:
                new_state[index] = new_
        else:
            new_ = int(new_state[index] * 1.1)
            if new_ == new_state[index]:
                new_state[index] = new_ + 1
            else:
                new_state[index] = new_
        if new_state[index] in Interval(argument_range[index][0], argument_range[index][1]):
            return new_state, True
        return before, False

    def step(self, action):
        if action in (0, 2, 4, 6, 8, 10):
            new_state, reward = self.get_effect_of_this(argument_name[int(action/2)], True)
        elif action in (1, 3, 5, 7, 9, 11):
            new_state, reward = self.get_effect_of_this(argument_name[int((action-1)/2)], False)
        else:
            raise Exception('wrong action')
        return new_state, reward, False, {}

    def reset(self):
        database_conn = self.database_conn
        time_before = self.query_test(database_conn)
        self.first_query_cost = time_before
        cur = database_conn.cursor()

        old_isolation_level = database_conn.isolation_level
        database_conn.set_isolation_level(0)
        cur.execute("""
                    ALTER SYSTEM SET maintenance_work_mem=65536;
        """)
        cur.execute("""
                    ALTER SYSTEM SET checkpoint_timeout=300;
                """)
        cur.execute("""
                    ALTER SYSTEM SET bgwriter_delay=200;
                """)
        cur.execute("""
                    ALTER SYSTEM SET bgwriter_lru_maxpages=100;
                """)
        cur.execute("""
                    ALTER SYSTEM SET bgwriter_lru_multiplier=2;
                """)
        cur.execute("""
                    ALTER SYSTEM SET work_mem=4096;
                """)
        cur.execute('select pg_reload_conf();')
        database_conn.set_isolation_level(old_isolation_level)
        self.state = torch.Tensor(default_argument)
        return self.state, {}

    def data_compute(self):
        #  一千万数据大概1350s左右
        fake = Faker(locale='zh_CN')
        database_conn = self.database_conn
        cur = database_conn.cursor()
        t1 = time.time()
        for i in range(1000):
            s = ''
            for j in range(1, 10000 + 1):
                s += '\t'.join([str(i * 10000 + j),
                                fake.name(),
                                str(random.randint(18, 48)),
                                fake.city_name(),
                                fake.phone_number(),
                                str(random.randint(5000, 50000)),
                                fake.job(),
                                str(fake.unix_time(end_datetime=None, start_datetime=None)),
                                str(random.randint(1, 9)),
                                ])
                s += '\n'
            cur.copy_from(StringIO(s), 'ppo_data',
                          columns=('id', 'name', 'age', 'address', 'telephone',
                                   'salary', 'section', 'emp_id', 'section_id'))
            database_conn.commit()
        database_conn.close()


if __name__ == '__main__':
    t1 = time.time()
    ss = PostGreEnv()
    first = ss.reset()
    print(first)
    ss.step(1)
    print(time.time() - t1)

