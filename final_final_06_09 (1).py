import statistics
import pandas as pd
import random
import numpy as np
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

def handleDataset(Location):                                                   # 資料預處理

    df = pd.read_csv(Location, header = None)
    df = df.fillna(0)
    
    return df

def standardization(data):                                                     # 資料標準化
    list_data = data.values.tolist()

    average_all = []
    std = []
    for x in range(len(list_data[0])):
        temp_1 = []
        for y in list_data:
            temp_1.append(y[x])
        std.append(statistics.pstdev(temp_1))
        temp_2 = sum(temp_1) / len(list_data)
        average_all.append(temp_2)
    for x in range(len(list_data[0])):
        for y in list_data:
            if std[x] != 0:
                y[x] = (y[x] - average_all[x]) / std[x]   
    data = pd.DataFrame(list_data)
    return data
                
def outlier(data):                                                             # 蓋帽法處理 outlier
    list_data = data.values.tolist()
    for n in list_data:
        for m in range(len(n)):   
            if abs(n[m]) > 3:
                if n[m] >= 0:
                    n[m] = 3
                else:
                    n[m] = -3
    data = pd.DataFrame(list_data)
    return data

def e_distance(data_1 , data_2):                                               # 計算歐式距離
    squared_diff = []
    for n in range(len(data_1)):
        squared_diff.append((data_1[n] - data_2[n]) ** 2)   
    sum_squared_diff = sum(squared_diff)    
    return sum_squared_diff ** 0.5

def clustering_Known(df_data, df_label, start_pos):                                       # 已知類分群
    group = max(df_label)[0]
    labels = []
    indexs = []
    for x in range(group):
        labels.append([])
        indexs.append([])

    for x in range(start_pos, group):
        for y in range(len(df_label)):
            if df_label[y][0] == (x + 1):
                labels[x].append(df_data[y])
                indexs[x].append(y)
    return labels, indexs

def centroid(labels):                                                          # 計算形心點
    means_points = []
    for label in labels:
        if len(label) != 1:
            means_point = []
            for y in range(len(label[0])):
                temp = []
                for x in range(len(label)):
                    temp.append(label[x][y])
                means_point.append(sum(temp)/len(temp))
            means_points.append(means_point)
        else:
            means_points.append(label[0]) 
    return means_points

def Range(labels, centroids):                                                  # 已知類可接受範圍
    Range = []
    fourStd = []
    for x in range(len(labels)): 
        if len(labels[x]) > 1:
            dist_centroid = []
            for y in range(len(labels[x])):
                dist = e_distance(labels[x][y], centroids[x])
                dist_centroid.append(dist)    
            std = statistics.pstdev(dist_centroid)
            fourStd.append(4 * std)
            Range.append(4 * std)
        else:
            Range.append(0)
    
    for x in range(len(Range)):
        if Range[x] == 0:
            Range[x] = min(fourStd) / 4
    print("形心半徑：")
    print(Range)
    return Range

def unknown_decide(dist_centroid, Ranges, index):
    unknown_index = []
    known_index = []
    for x in range(len(dist_centroid)):
        for y in range(len(dist_centroid[x])):
            if dist_centroid[x][y] >= Ranges[x]:
                unknown_index.append(index[x][y])
            else:
                known_index.append(index[x][y])
    return unknown_index, known_index

def cut_testData(index, df_data_test, df_label_test):
    data = []
    label = []
    for x in range(len(index)):
        y = index[x]
        data.append(df_data_test[y])
        label.append(df_label_test[y][0])
    return data, label
        
def classfication(df_data, df_label, df_data_test, df_label_test, pred_label_test):
    
    df_data = df_data.values.tolist()
    df_label = df_label.values.tolist()
    df_data_test = df_data_test.values.tolist()
    df_label_test = df_label_test.values.tolist()

    pred_label_test = pred_label_test.tolist()
    pred_label_test = pd.DataFrame(pred_label_test)
    pred_label_test = pred_label_test.values.tolist()

    labels_known, indexs_known = clustering_Known(df_data, df_label, 0)
    centroids = centroid(labels_known)
    Ranges = Range(labels_known, centroids)

    pred_labels_known, pred_indexs_known = clustering_Known(df_data_test, pred_label_test, 0)

    dist_centroids = []
    for x in range(len(pred_labels_known)):
        dist_centroid = []
        for y in range(len(pred_labels_known[x])):
            dist = e_distance(pred_labels_known[x][y], centroids[x])
            dist_centroid.append(dist)
        dist_centroids.append(dist_centroid)

    unknown_index, known_index = \
        unknown_decide(dist_centroids, Ranges, pred_indexs_known)

    unknown_df_data_test, unknown_df_label_test = \
        cut_testData(unknown_index, df_data_test, df_label_test)
    known_df_data_test, known_df_label_test = \
        cut_testData(known_index, df_data_test, df_label_test)

    return unknown_df_data_test, unknown_df_label_test, known_df_data_test, known_df_label_test

# K-means
def create_centroid(k, unknown_data):                                                          # create random centroids

    centroids = random.sample(unknown_data, k)

    return centroids

def assign_data(centroids, unknown_data, k):                                                   # cluster data with centroids
    cluster = {}
    cluster_index = {}
    for i in range(k):
        cluster[i] = []
        cluster_index[i] = []
    for x in range(len(unknown_data)):
        temp_dist = []
        for i in range(len(centroids)):
            # get closest centroid distance -> cluster
            temp_dist.append(e_distance(centroids[i], unknown_data[x]))
        closet_centroid_index = temp_dist.index(min(temp_dist)) # get closet centroid index

        cluster[closet_centroid_index].append(unknown_data[x])
        cluster_index[closet_centroid_index].append(x)

    print(f"0: {len(cluster[0])}, 1: {len(cluster[1])}, 2: {len(cluster[2])}, 3: {len(cluster[3])}, 4: {len(cluster[4])}")
    # print(pd.DataFrame(cluster[0]))
    return cluster, cluster_index

def index_to_label(cluster_index, unknown_label):                                                # cluster label
    cluster_index = list(cluster_index.values())
    cluster_unknown_label = []

    for i in range(len(cluster_index)):
        for x in range(len(cluster_index[i])):
            cluster_index[i][x] = unknown_label[cluster_index[i][x]]

    return cluster_index

def detect_stop(new_centroids, prev_centroids, k):                                                # kmeans stop criterion

    for i in range(k):
        centroids_shift = e_distance(new_centroids[i], prev_centroids[i])

        if centroids_shift < 0.000000000000001:
            return True

    return False


def kmeans(unknown_data, unknown_label,k):                                                        # do kmeans

    prev_centroids= create_centroid(k, unknown_data)
    cluster, cluster_index = assign_data(prev_centroids, unknown_data, k)  # return a cluster
    list_cluster_label = index_to_label(cluster_index, unknown_label) # return cluster label
    list_cluster = list(cluster.values())
    new_centroids = centroid(list_cluster)
    # print(f"prev centroids: \n{pd.DataFrame(new_centroids)}")

    is_clustering = True
    while is_clustering:
        prev_centroids = new_centroids
        cluster, cluster_index = assign_data(prev_centroids, unknown_data, k)  # return a cluster
        list_cluster_label = index_to_label(cluster_index, unknown_label)  # return cluster label
        list_cluster = list(cluster.values())
        new_centroids = centroid(list_cluster)

        if detect_stop(new_centroids, prev_centroids, k):
            is_clustering = False

    return list_cluster, list_cluster_label, new_centroids

# ------------ #
# after kmeans
def get_origin_index(origin_all_distances, min_min_dist):

    # create dict
    all_dist = {}
    for x in range(len(origin_all_distances)):
        all_dist[x] = []
    for x in range(len(origin_all_distances)):
        all_dist[x] = origin_all_distances[x]

    # find value
    for x in range(len(all_dist)):
        if min_min_dist in all_dist[x]:
            get_key = x
            return get_key
    # return key

def get_min(all_distances):
    closest_index = []
    current_index = []
    origin_all_distances = copy.deepcopy(all_distances)

    for i in range(len(all_distances)):
        min_dist = []
        for x in range(len(all_distances)):

            min_dist.append(min(i for i in all_distances[x] if i != -1))

        closest_index.append(get_origin_index(origin_all_distances, min(min_dist)))
        current_index.append(min_dist.index(min(min_dist)))


        for x in range(len(all_distances)):
            all_distances[x][current_index[i]] = -1

        all_distances.pop(current_index[i])

    return closest_index

def get_accuracy(predict_label, cluster_label, closet_centroid_index):
    error_rate = []
    for x in range(len(closet_centroid_index)):
        current_check = predict_label[closet_centroid_index[x]][0]
        # print(f"current_check: {current_check}")
        error_num = 0
        for i in range(len(cluster_label[x])):
            if cluster_label[x][i] != current_check:
                error_num += 1
        error_rate.append(error_num/len(cluster_label[x]))
        print(f"cluster {x} error rate: {error_num/len(cluster_label[x])}")

    accuracy = 1 - (sum(error_rate) / len(error_rate))

    print(f"分群結果accuracy: {accuracy}")


def main():

    #''' part2
    data_location = '/Users/kimmy_yo/Downloads/train_data (1).csv'
    label_location = '/Users/kimmy_yo/Downloads/train_label.csv'
    test_data_location = '/Users/kimmy_yo/Downloads/test_data (1).csv'
    test_label__location = '/Users/kimmy_yo/Downloads/test_label.csv'
    df_data = handleDataset(data_location)
    df_label = handleDataset(label_location)
    df_data_test = handleDataset(test_data_location)
    df_label_test = handleDataset(test_label__location)
    
    df_data = standardization(df_data)
    df_data_test = standardization(df_data_test)
    df_data = outlier(df_data)
    df_data_test = outlier(df_data_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(df_data, df_label.values.ravel())
    pred_knn = knn.predict(df_data_test)


    unknown_df_data_test, unknown_df_label_test, known_df_data_test, known_df_label_test = \
        classfication(df_data, df_label, df_data_test, df_label_test, pred_knn)          # 分類

    df_data_test = df_data_test.values.tolist()
    df_label_test = df_label_test.values.tolist()


    cluster, cluster_label, new_centroids = kmeans(unknown_df_data_test, unknown_df_label_test, k=5)
    for cluster in cluster_label:
        print(f"clusters: \n{cluster}")
    labels_unknown, index_unknown = clustering_Known(df_data_test, df_label_test, 8)
    labels_unknown = labels_unknown[8:]
    index_unknown = index_unknown[8:]

    test_centroids = centroid(labels_unknown)

    # print(pd.DataFrame(test_centroids))
    # print(pd.DataFrame(new_centroids))

    all_distances = []
    for i in range(len(new_centroids)):
        distances = []
        for j in range(len(test_centroids)):
            dist = e_distance(new_centroids[i], test_centroids[j])
            distances.append(dist)

        all_distances.append(distances)

    closet_centroid_index = get_min(all_distances)  # new_centroid 靠近第幾個 test_centroid
    # print(f"{closet_centroid_index}")


    predict_label = []
    for x in range(len(closet_centroid_index)):
        predict_label.append(df_label_test[index_unknown[closet_centroid_index[x]][0]])  # get 靠近第幾個 test_centroid 的資料

    print(predict_label)

    get_accuracy(predict_label, cluster_label, closet_centroid_index)

    # print("--------------------------------------------------")
    # print("被歸類於未知類別之資料的正確標籤：")
    # print(unknown_df_label_test)
    # print("共" + str(len(unknown_df_label_test)) + "筆")
    # print("被歸類於已知類別之資料的正確標籤：")
    # print(known_df_label_test)
    # print("共" + str(len(known_df_label_test)) + "筆")
    # print("--------------------------------------------------")
    #
    # knn = KNeighborsClassifier(n_neighbors = 1)
    # knn.fit(df_data, df_label)
    # pred_knn = knn.predict(known_df_data_test)
    # print("使用 knn 分類已知類別之資料的預測標籤：")
    # print(pred_knn)
    # print("混淆矩陣：")
    # print(confusion_matrix(known_df_label_test, pred_knn))
    # print(classification_report(known_df_label_test, pred_knn))
    # print("--------------------------------------------------")
    # tree = DecisionTreeClassifier()
    # tree.fit(df_data, df_label)
    # pred_tree = tree.predict(known_df_data_test)
    # print("使用分類樹分類已知類別之資料的預測標籤：")
    # print(pred_tree)
    # print("混淆矩陣：")
    # print(confusion_matrix(known_df_label_test, pred_tree))
    # print(classification_report(known_df_label_test, pred_tree))
    #
    #'''    
main()