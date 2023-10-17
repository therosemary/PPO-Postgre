import time
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import psycopg2
import torch
from faker import Faker
from interval3 import Interval

from sql_str import *

plt.style.use('_mpl-gallery')

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
        self.learning_data_before = []
        self.learning_data_after = [[], [], []]

    @staticmethod
    def query_test(database_conn):
        """单表随机读性能查询"""
        s = time.time()
        cur = database_conn.cursor()
        for i in range(1):
            cur.execute(sing_query_sql)
        cost = time.time() - s
        return cost/1

    @staticmethod
    def query_inner_test(database_conn):
        """多表联查性能测试"""
        s = time.time()
        cur = database_conn.cursor()
        for i in range(1):
            cur.execute(multi_query_sql)
        # res = cur.fetchall()
        cost = time.time() - s
        return cost/1

    @staticmethod
    def update_test(database_conn):
        """更新表操作"""
        s = time.time()
        cur = database_conn.cursor()
        for i in range(1):
            cur.execute(update_query_sql)
        # res = cur.fetchall()
        cost = time.time() - s
        return cost/1

    def calculate_cost(self, database_conn, is_save=False):
        query1 = self.query_test(database_conn)
        query2 = self.query_inner_test(database_conn)
        query3 = self.update_test(database_conn)
        if is_save:
            self.learning_data_after[0].append(query1)
            self.learning_data_after[1].append(query2)
            self.learning_data_after[2].append(query3)
        return query1 + query2 + query3

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
            time_before = self.calculate_cost(database_conn)
            alter_sql = self.get_alter_sql(argument, new_state[argument_name_dict[argument]])
            reload_sql = 'select pg_reload_conf();'
            cur.execute(alter_sql)
            cur.execute(reload_sql)

            database_conn.set_isolation_level(old_isolation_level)
            cur.execute('show shared_buffers;')
            time_after = self.calculate_cost(database_conn)
            time_change = time_after - time_before
            self.query_cost_now = time_after
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

    def show_plt(self):
        plt_data = self.learning_data_after
        x = np.arange(0, len(plt_data[0]))
        y = np.vstack([plt_data[0], plt_data[1], plt_data[2]])

        # plot
        fig, ax = plt.subplots()

        ax.stackplot(x, y)

        # ax.set(xlim=(0, 8), xticks=np.arange(1, 8),
        #        ylim=(0, 8), yticks=np.arange(1, 8))

        plt.show()

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
        # for i in range(1000):
        #     s = ''
        #     for j in range(1, 10000 + 1):
        #         s += '\t'.join([str(i * 10000 + j),
        #                         fake.name(),
        #                         str(random.randint(18, 48)),
        #                         fake.city_name(),
        #                         fake.phone_number(),
        #                         str(random.randint(5000, 50000)),
        #                         fake.job(),
        #                         str(fake.unix_time(end_datetime=None, start_datetime=None)),
        #                         str(random.randint(1, 9)),
        #                         ])
        #         s += '\n'
        #     cur.copy_from(StringIO(s), 'ppo_data',
        #                   columns=('id', 'name', 'age', 'address', 'telephone',
        #                            'salary', 'section', 'emp_id', 'section_id'))
        #     database_conn.commit()
        for i in range(100):
            s = ''
            for j in range(1, 100 + 1):
                s += '\t'.join([str(i * 100 + j),
                                str(random.randint(1, 9)),
                                fake.name(),
                                str(random.randint(1, 100)),
                                fake.city_name(),
                                fake.color(),
                                str(random.randint(1, 33)),
                                ])
                s += '\n'
            cur.copy_from(StringIO(s), 'ppo_section',
                          columns=('id', 'section_id', 'name', 'priority', 'address', 'color',
                                   'floor'))
            database_conn.commit()
        database_conn.close()


if __name__ == '__main__':
    t1 = time.time()
    ss = PostGreEnv()
    fake = Faker(locale='zh_CN')


