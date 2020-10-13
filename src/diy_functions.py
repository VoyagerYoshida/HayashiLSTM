import numpy as np


def produce_canditates(df_data, df_test_ids, TARGET_IDX):
    cans_vecs = []
    candi_ids = df_test_ids.iloc[:,1:]
    for i in range(len(candi_ids)):
        can_line = candi_ids.iloc[i,:].values

        vec_full_0 = df[df.set_id.isin([can_line[0]])].vec.values
        vec_full_1 = df[df.set_id.isin([can_line[1]])].vec.values
        vec_full_dummy = df[df.set_id.isin([can_line[2]])].vec.values
        vec_all = []
        vec_all.append(vec_full_0)
        vec_all.append(vec_full_1)
        vec_all.append(vec_full_dummy)
        vec_all = np.array(vec_all)

        can_vec0 = []
        can_vec0.append(vec_all[0][0][TARGET_IDX])
        can_vec0.append(vec_all[1][0][TARGET_IDX])
        can_vec0.append(vec_all[2][0][TARGET_IDX])
        can_vec0 = np.array(can_vec0)

        cans_vecs.append(can_vec0)
    cans_vecs = np.array(cans_vecs)
    return cans_vecs


def accuracy_quiz(y_pred, y_test, y_candis):
    rank_points = []
    pinpon = 0
    for i in range(len(y_candis)):
        distance = []
        distance.append(np.linalg.norm(y_test[i] - y_pred[i]))
        for j in range(len(y_candis[0])):
            distance.append(np.linalg.norm(y_candis[i][j] - y_pred[i]))
        distance = np.array(distance)
        rank = np.argsort(distance)
        rank_points.append(rank_point(rank))
        if rank[0]==0:
            pinpon += 1
    acc = pinpon / len(y_candis) * 100
    rank_points = np.array(rank_points)
    mean_rank = rank_points.mean()
    return acc, mean_rank


def rank_point(rank):
    if rank[0] == 0:
        return 1
    elif rank[0] == 1:
        return 0.5
    elif rank[0] == 2:
        return 0.25
    elif rank[0] == 3:
        return 0

