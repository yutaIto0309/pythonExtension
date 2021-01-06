import pandas as pd
#%%
# 顧客情報
customer_master = pd.read_csv('/Users/itouyuuta/Desktop/Python/pythonStudies/python_100_knocks/1章/customer_master.csv')
#customer_master.head()

# 商品情報
item_master = pd.read_csv('/Users/itouyuuta/Desktop/Python/pythonStudies/python_100_knocks/1章/item_master.csv')
#item_master.head()

# 売上明細1
transaction_1 = pd.read_csv('/Users/itouyuuta/Desktop/Python/pythonStudies/python_100_knocks/1章/transaction_1.csv')
#transaction_1.head()

# 売上明細詳細1
transaction_detail_1 = pd.read_csv('/Users/itouyuuta/Desktop/Python/pythonStudies/python_100_knocks/1章/transaction_detail_1.csv')
#transaction_detail_1.head()

# 売上明細1と2の結合
transaction_2 = pd.read_csv('/Users/itouyuuta/Desktop/Python/pythonStudies/python_100_knocks/1章/transaction_2.csv')
transaction = pd.concat([transaction_1, transaction_2], ignore_index=True)

# 売上明細詳細1と2の結合
transaction_detail_2 = pd.read_csv('/Users/itouyuuta/Desktop/Python/pythonStudies/python_100_knocks/1章/transaction_detail_2.csv')
transaction_detail = pd.concat([transaction_detail_1, transaction_detail_2], ignore_index=True)
# 詳細データに明細を結合する。
join_data = pd.merge(transaction_detail, transaction[['transaction_id', 'payment_date', 'customer_id']], on='transaction_id', how='left')
join_data.head()


