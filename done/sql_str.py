import random



get_argument_sql = """
    SELECT name, setting, unit,context FROM pg_settings WHERE name like '%buffer%';
    """
create_table1_sql = """
    CREATE TABLE ppo_data(
       id             INTEGER    NOT NULL PRIMARY KEY,
       name           CHAR(50)                NOT NULL,
       age            INTEGER                 NOT NULL,
       address        CHAR(50)         NOT NULL,
       salary         INTEGER                 NOT NULL,
       section        CHAR(50)         NOT NULL,
       emp_id         CHAR(50)         NOT NULL,
       section_id     INTEGER                 NOT NULL);
    """
create_table2_sql = """
        CREATE TABLE ppo_section(
           id             INTEGER                   NOT NULL PRIMARY KEY,
           name           CHAR(50)               NOT NULL,
           priority       INTEGER                   NOT NULL,
           address        CHAR(50)               NOT NULL,
           color          CHAR(50)               NOT NULL,
           floor          INTEGER                   NOT NULL);
        """
query_sql = """
    SELECT id,name,address FROM ppo_data WHERE ID IN({},{},{},{})
    
    """.format(random.randint(1, 10000000),
               random.randint(1, 10000000),
               random.randint(1, 10000000),
               random.randint(1, 10000000))
