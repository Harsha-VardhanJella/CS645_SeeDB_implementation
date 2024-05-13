import pandas as pd
import psycopg2
import numpy as np
from scipy.special import kl_div
import matplotlib.pyplot as plt
from sklearn.base import defaultdict

#db-connection variables
hostname='localhost'
database='postgres'
username='postgres'
pwd='harsha$23'
port_id=5434
adult_path="census+income/adult.data"

db_conn = psycopg2.connect( host=hostname,dbname=database,user=username,password=pwd,port=port_id)
db_cursor = db_conn.cursor()


##Combining Target and reference query 
def shared_tr(f,m,a,db_conn):
    db_cursor = db_conn.cursor()
    combined_query=f"SELECT {a},{f}({m}),CASE WHEN marital_status IN (' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent',' Separated') THEN 1 ELSE 0 END AS g1,CASE WHEN marital_status IN (' Never-married', ' Widowed',' Divorced') THEN 1 ELSE 0 END AS g2 FROM census_data GROUP BY {a},g1,g2;"
    db_cursor.execute(combined_query)
    result=db_cursor.fetchall()
    plot_tr(result,a,f,m)

def Error(m, N):
    delta=0.1
    return np.sqrt((1 - (m - 1) / N) * (2 * np.log(np.log(m)) + np.log(np.pi**2 / (3 * delta)))/ (2 * m))


#Plotting Both Target and reference in the same graph
def plot_tr(result_array,a,f,m):

    # Extracting data for the first graph
    categories = list(set(row[0] for row in result_array))
    categories = sorted(categories, reverse=True)
    values_g1 = [0] * len(categories)
    values_g2 = [0] * len(categories)
    for row in result_array:
        category_index = categories.index(row[0])
        if row[2] == 1:
            values_g1[category_index] = row[1]
        else:
            values_g2[category_index] = row[1]
    bar_width = 0.35
    x = np.arange(len(categories))
    plt.bar(x - bar_width/2, values_g1, bar_width, color='blue', label='married')
    plt.bar(x + bar_width/2, values_g2, bar_width, color='orange', label='unmarried')
    plt.yscale('log')
    plt.xlabel(a)
    plt.ylabel(f+"("+m+")")
    plt.title('Comparison of Values by Category')
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

#Plotting individual graphs
def plot(f,m,a,db_conn):
    cur=db_conn.cursor()
    query1=f"select {a},{f}({m}) from married group by {a}"
    cur.execute(query1)

    rows = cur.fetchall()
    a_values = [row[0] for row in rows]
    f_values = [row[1] for row in rows]
    fig, ax = plt.subplots()

    plt.bar(a_values, f_values)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, ha="right" )
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.xlabel(a)
    plt.ylabel(m+"-"+f)
    plt.title("Married Bar Graph")
    plt.show()

    query2=f"select {a},{f}({m}) from unmarried group by {a}"
    cur.execute(query2)

    rows1 = cur.fetchall()
    cur.close()
    a_values1 = [row[0] for row in rows1]
    f_values1 = [row[1] for row in rows1]
    fig, ax = plt.subplots()
    plt.bar(a_values1, f_values1)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=90, ha="right" )
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel(a)
    plt.ylabel(m+f)
    plt.title("Unmarried Bar Graph")
    plt.show()
    

def kl(p1,p2):
    x=kl_div(p1,p2)
    return x

def initial(check_columns,measures,Agg_func,db_conn,db_cursor):
    utility=[]
    utility_a=[]
    utility_m=[]
    utility_f=[]
    for a in check_columns:
        for m in measures:
            for f in Agg_func:
                query1=f"select {f}({m}) from married group by {a}"
                query2=f"select {f}({m}) from unmarried group by {a}"
                db_cursor.execute(query1)
                result1=db_cursor.fetchall()
                result1=np.array(result1).flatten()
                normalized_r1 = (result1 - np.min(result1)) / (np.max(result1) - np.min(result1))
                normalized_r1[np.isinf(normalized_r1) | np.isnan(normalized_r1)] = 0
                db_cursor.execute(query2)
                result2=db_cursor.fetchall()
                result2=np.array(result2).flatten()
                normalized_r2 = (result2 - np.min(result2)) / (np.max(result2) - np.min(result2))
                normalized_r2[np.isinf(normalized_r2) | np.isnan(normalized_r2)] = 0
                n1=np.array(normalized_r1).flatten()
                n2=np.array(normalized_r2).flatten()
                if len(n1)==len(n2):
                    x =kl(n1,n2)
                else:
                    if len(n1) > len(n2):
                        n1 = np.random.choice(n1, len(n2))
                    elif len(n2) > len(n1):
                        n2 = np.random.choice(n2, len(n1))
                    x=kl(n1,n2)
                x[np.isinf(x) | np.isnan(x)] = 0
                utility.append(np.sum(x))
                utility_a.append(a)
                utility_m.append(m)
                utility_f.append(f)
    utility=np.array(utility)
    utility[np.isinf(utility) | np.isnan(utility)] = 0
    
    script_create_utility="""create table if not exists utility(utility_score real,M text,A text,F text) """
    db_cursor.execute(script_create_utility)
    #insertion to utility 
    for i in range(len(utility)):
        script_insert_utility= f"insert into utility (utility_score, M, A, F) values ({utility[i]}, '{utility_m[i]}', '{utility_a[i]}', '{utility_f[i]}')"
        db_cursor.execute(script_insert_utility)

    #view_script="""SELECT * FROM utility where m='capital_gain' and a='sex' and f='avg' order by utility_score desc LIMIT 5"""
    view_script="""SELECT * FROM utility order by utility_score desc LIMIT 5"""

    db_cursor.execute(view_script)
    result_test=db_cursor.fetchall()
    for i in range(5):
        _,m,a,f=result_test[i]
        plot(f,m,a,db_conn)
        shared_tr(f,m,a,db_conn)

#combining multiple agrregates
def shared_op1(check_columns,measures,Agg_func,db_conn,db_cursor):
    util_score=[]
    util_score_a=[]
    util_score_m=[]
    util_score_f=[]
    sql_queries1 = []
    sql_queries2 = []    
    #combined multiple aggregates
    for col in check_columns:
        for measure in measures:
            sql_query1 = f"SELECT "
            sql_query2 = f"SELECT "
            # Add aggregation functions for the current measure
            for func in Agg_func:
                sql_query1+= f"{func}({measure}), "
                sql_query2+= f"{func}({measure}), "
            # Add the single group by column
            sql_query1 += f"{col} FROM married GROUP BY {col}"
            sql_query2 += f"{col} FROM unmarried GROUP BY {col}"
            
            db_cursor.execute(sql_query1)
            result1=db_cursor.fetchall()
            result1=np.array(result1)
            f_r1=result1.T
            f_r1=f_r1[:-1].astype(float)
            normalized_r1 = (f_r1 - np.min(f_r1,axis=1,keepdims=True)) / (np.max(f_r1,axis=1,keepdims=True) - np.min(f_r1,axis=1,keepdims=True))
            normalized_r1[np.isinf(normalized_r1) | np.isnan(normalized_r1)] = 0
            db_cursor.execute(sql_query2)
            result2=db_cursor.fetchall()
            result2=np.array(result2)
            f_r2=result2.T
            f_r2=f_r2[:-1].astype(float)
            normalized_r2 = (f_r2 - np.min(f_r2,axis=1,keepdims=True)) / (np.max(f_r2,axis=1,keepdims=True) - np.min(f_r2,axis=1,keepdims=True))
            normalized_r2[np.isinf(normalized_r2) | np.isnan(normalized_r2)] = 0
            for i in range(normalized_r1.shape[0]):
                n1=np.array(normalized_r1[i]).flatten()
                n2=np.array(normalized_r2[i]).flatten()
                if len(n1)==len(n2):
                    x =kl(n1,n2)
                else:
                    if len(n1) > len(n2):
                        n1 = np.random.choice(n1, len(n2))
                    elif len(n2) > len(n1):
                        n2 = np.random.choice(n2, len(n1))
                    x=kl(n1,n2)
                x[np.isinf(x) | np.isnan(x)] = 0
                util_score.append(np.sum(x))
                util_score_a.append(col)
                util_score_m.append(measure)
                util_score_f.append(Agg_func[i])
                sql_queries1.append(sql_query1)
                sql_queries2.append(sql_query2)
    util_score=np.array(util_score)
    util_score[np.isinf(util_score) | np.isnan(util_score)] = 0
    script_create_utility="""create table if not exists util_score_cAttr(utility_score real,M text,A text,F text) """
    db_cursor.execute(script_create_utility)
    #insertion to utility 
    '''for i in range(len(util_score)):
        script_insert_utility= f"insert into util_score_cAttr (utility_score, M, A, F) values ({util_score[i]}, '{util_score_m[i]}', '{util_score_a[i]}', '{util_score_f[i]}')"
        db_cursor.execute(script_insert_utility)'''

    #view_script="""SELECT * FROM utility where m='capital_gain' and a='sex' and f='avg' order by utility_score desc LIMIT 5"""
    view_script="""SELECT * FROM util_score_cAttr order by utility_score desc LIMIT 5"""
    db_cursor.execute(view_script)
    result_test=db_cursor.fetchall()
    for i in range(5):
        _,m,a,f=result_test[i]
        print("Generic Plots")
        plot(f,m,a,db_conn)
        print("combined-plots")
        shared_tr(f,m,a,db_conn)
#combined reference target and reference
def shared_op2(check_columns,measures,Agg_func,db_conn,db_cursor):
    util_score1=[]
    util_score_a1=[]
    util_score_m1=[]
    util_score_f1=[]
    for a in check_columns:
        for m in measures:
            for f in Agg_func:
                com_query=f"SELECT {a},{f}({m}) AS f_m,CASE WHEN marital_status IN (' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent',' Separated') THEN 1 ELSE 0 END AS g1,CASE WHEN marital_status IN (' Never-married', ' Widowed',' Divorced') THEN 1 ELSE 0 END AS g2 FROM census_data GROUP BY {a}, g1, g2;"
                cur=db_conn.cursor()
                cur.execute(com_query)
                result1 = []
                result2 = []
                result=cur.fetchall()
                for row in result:
                    if row[2] == 1:
                        result1.append(row[1])
                    elif row[3] == 1:
                        result2.append(row[1])
                result1=np.array(result1).flatten()
                normalized_r1 = (result1 - np.min(result1,keepdims=True)) / (np.max(result1,keepdims=True) - np.min(result1,keepdims=True))
                normalized_r1[np.isinf(normalized_r1) | np.isnan(normalized_r1)] = 0
                result2=np.array(result2).flatten()
                normalized_r2 = (result2 - np.min(result2)) / (np.max(result2) - np.min(result2))
                normalized_r2[np.isinf(normalized_r2) | np.isnan(normalized_r2)] = 0
                n1=np.array(normalized_r1).flatten()
                n2=np.array(normalized_r2).flatten()
                if len(n1)==len(n2):
                    x =kl(n1,n2)
                else:
                    if len(n1) > len(n2):
                        n1 = np.random.choice(n1, len(n2))
                    elif len(n2) > len(n1):
                        n2 = np.random.choice(n2, len(n1))
                    x=kl(n1,n2)
                x[np.isinf(x) | np.isnan(x)] = 0
                util_score1.append(np.sum(x))
                util_score_a1.append(a)
                util_score_m1.append(m)
                util_score_f1.append(f)
                
    util_score1=np.array(util_score1)
    util_score1[np.isinf(util_score1) | np.isnan(util_score1)] = 0
    
    script_create_utility="""create table if not exists util_tr(utility_score real,M text,A text,F text) """
    db_cursor.execute(script_create_utility)
    db_cursor.execute(script_create_utility)
    #insertion to utility 
    for i in range(len(util_score1)):
        script_insert_utility= f"insert into util_score_cAttr (utility_score, M, A, F) values ({util_score1[i]}, '{util_score_m1[i]}', '{util_score_a1[i]}', '{util_score_f1[i]}')"
        db_cursor.execute(script_insert_utility)

    #view_script="""SELECT * FROM utility where m='capital_gain' and a='sex' and f='avg' order by utility_score desc LIMIT 5"""
    view_script="""SELECT * FROM util_score_cAttr order by utility_score desc LIMIT 5"""
    db_cursor.execute(view_script)
    result_test=db_cursor.fetchall()
    for i in range(5):
        _,m,a,f=result_test[i]
        print("Generic Plots")
        plot(f,m,a,db_conn)
        print("combined-plots")
        shared_tr(f,m,a,db_conn)

def shared_op3(check_columns,measures,Agg_func,db_conn,db_cursor,dbname):
    util_score2=[]
    util_score_a2=[]
    util_score_m2=[]
    util_score_f2=[]
    sql_queries1 = []
    sql_queries2 = [] 
    db_name=dbname   
    #combined multiple aggregates
    for col in check_columns:
        for measure in measures:
            sql_query1 = f"SELECT "
            # Add aggregation functions for the current measure
            for func in Agg_func:
                sql_query1+= f"{func}({measure}), "
            # Add the single group by column
            sql_query1 += f"{col} ,CASE WHEN marital_status IN (' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent',' Separated') THEN 1 ELSE 0 END AS g1,CASE WHEN marital_status IN (' Never-married', ' Widowed',' Divorced') THEN 1 ELSE 0 END AS g2 FROM {db_name} GROUP BY {col}, g1, g2;"
            cur=db_conn.cursor()
            cur.execute(sql_query1)
            result_m = []
            result_um = []
            result=cur.fetchall()
            for row in result:
                if row[-2] == 1:
                    result_m.append(row)
                elif row[-1] == 1:
                    result_um.append(row)
            result1=np.array(result_m)

            f_r1=result1.T
            f_r1=f_r1[:-3].astype(float)
            prob_r1= f_r1 / np.sum(f_r1, axis=1, keepdims=True)

            '''normalized_r1 = (f_r1 - np.min(f_r1,axis=1,keepdims=True)) / (np.max(f_r1,axis=1,keepdims=True) - np.min(f_r1,axis=1,keepdims=True))
            normalized_r1[np.isinf(normalized_r1) | np.isnan(normalized_r1)] = 0'''
            result2=np.array(result_um)
            f_r2=result2.T
            f_r2=f_r2[:-3].astype(float)
            '''normalized_r2 = (f_r2 - np.min(f_r2,axis=1,keepdims=True)) / (np.max(f_r2,axis=1,keepdims=True) - np.min(f_r2,axis=1,keepdims=True))
            normalized_r2[np.isinf(normalized_r2) | np.isnan(normalized_r2)] = 0'''
            prob_r2=f_r2 / np.sum(f_r1, axis=1, keepdims=True)
            for i in range(prob_r1.shape[0]):
                n1=np.array(prob_r1[i]).flatten()
                n2=np.array(prob_r2[i]).flatten()
                if len(n1)==len(n2):
                    x =kl(n1,n2)
                else:
                    if len(n1) > len(n2):
                        n1 = np.random.choice(n1, len(n2))
                    elif len(n2) > len(n1):
                        n2 = np.random.choice(n2, len(n1))
                    x=kl(n1,n2)
                x[np.isinf(x) | np.isnan(x)] = 0
                util_score2.append(np.sum(x))
                util_score_a2.append(col)
                util_score_m2.append(measure)
                util_score_f2.append(Agg_func[i])
    util_score2=np.array(util_score2)
    util_score2[np.isinf(util_score2) | np.isnan(util_score2)] = 0
    script_create_utility=f"create table if not exists util_score_socombined1_{db_name}(utility_score real,M text,A text,F text) "
    db_cursor.execute(script_create_utility)
    #insertion to utility 
    '''for i in range(len(util_score2)):
        script_insert_utility= f"insert into util_score_socombined1_{db_name} (utility_score, M, A, F) values ({util_score2[i]}, '{util_score_m2[i]}', '{util_score_a2[i]}', '{util_score_f2[i]}')"
        db_cursor.execute(script_insert_utility)'''
    return util_score2

def prune(m, top_views, mean, N, k):
    if m == 1:
        return top_views
    else:
        ub={}
        lb={}
        error=Error(m, N)
        for q in top_views:
            ub[q] = mean[q] + error
            lb[q] = mean[q] - error
        top_k_views = np.array(list(ub.keys()))
        upper=np.array(list(ub.values()))
        sorted_indices=np.argsort(upper)[-k:]
        top_k_views=top_k_views[sorted_indices]
        lower_min=np.min(np.array(list(lb.values()))[sorted_indices])
        for query in top_views:
            if query not in top_k_views:
                if ub[query] < lower_min:
                    top_views.remove(query)
        return top_views
        
def custom_comparison(item):
    return item[1]

K=6 #number of splits in dataset 
T=5 #Top 5 views 
#census_data
adult_df= pd.read_csv(adult_path, sep=",")
#
check_columns = ["workclass" , "education" , "occupation" , "race" , "sex" , "economic_indicator", "relationship","native_country"]
measures = ["age","education_num","capital_gain","capital_loss","hours_per_week","fnlwgt"]
Agg_func=["max","min","avg","sum","count"]

mean_p = defaultdict(float)
for i in range (K):  
    db_name="db"+str(i+1)
    util_score=shared_op3(check_columns,measures,Agg_func,db_conn,db_cursor,db_name)
    view_script=f"select utility_score,a,m,f from util_score_socombined1_{db_name}"
    db_cursor.execute(view_script)
    views=db_cursor.fetchall()
    for view_key in views:
        mean_p[view_key]=(mean_p[view_key]*i+view_key[0])/(i+1)
    views=prune(i+1,views,mean_p,K,T)
top_k_items = sorted(mean_p.items(), reverse=True, key=custom_comparison)[:T]
print(top_k_items)
keys = [item[0] for item in top_k_items]
for i in range(5):
        _,a,m,f=keys[i]
        print("Generic Plots")
        plot(f,m,a,db_conn)
        print("combined-plots")
        shared_tr(f,m,a,db_conn)
db_conn.commit()
db_conn.close()