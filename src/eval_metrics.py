def clusters_to_classes(grouping, num_clusters):
    mapping = {}
    for i in range(num_clusters):
        if i in grouping.Clusters.unique():
            mapping[i] = grouping['PHDIAG'].count()[i].idxmax()
    return mapping


def clusters_to_binary(grouping, num_clusters):
    mapping = {}
    for i in range(num_clusters):
        if i in grouping.Clusters.unique():
            filt = mapping[i] = grouping['PHDIAG'].count()[i]
            ca = filt[filt.index != 'Annet'].sum()
            non = filt[filt.index == 'Annet'].sum()

            mapping[i] = 'non-CA' if ca < non else 'CA'

    return mapping


def eval_external_metrics(data, y, y_ca, num_k):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    c2c = clusters_to_classes(data.groupby(['Clusters', 'PHDIAG']), num_k)
    c2c_binary = clusters_to_binary(
        data.groupby(['Clusters', 'PHDIAG']), num_k)

    data['c2c'] = data['Clusters'].map(c2c)
    data['c2c_binary'] = data['Clusters'].map(c2c_binary)

    acc = accuracy_score(y, data['c2c'])
    rec = recall_score(y, data['c2c'], average='weighted')
    prec = precision_score(y, data['c2c'], average='weighted')
    f1 = f1_score(y, data['c2c'], average='weighted')

    acc_b = accuracy_score(y_ca, data['c2c_binary'])
    rec_b = recall_score(y_ca, data['c2c_binary'], average='weighted')
    prec_b = precision_score(y_ca, data['c2c_binary'], average='weighted')
    f1_b = f1_score(y_ca, data['c2c_binary'], average='weighted')

    print('Accuracy Score: %.3f' % acc)
    print('Recall Score: %.3f' % rec)
    print('Precision Score: %.3f' % prec)
    print('F1 Score: %.3f' % f1)

    print('CA/non-CA')
    print('Accuracy Score : %.3f' % acc_b)
    print('Recall Score: %.3f' % rec_b)
    print('Precision Score: %.3f' % prec_b)
    print('F1 Score : %.3f' % f1_b)

    return acc, rec, prec, f1, acc_b, rec_b, prec_b, f1_b


def eval_internal_metrics(X_preprocessed, pred):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    score_kemans_s = silhouette_score(X_preprocessed, pred, metric='euclidean')
    score_kemans_c = calinski_harabasz_score(X_preprocessed, pred)
    score_kemans_d = davies_bouldin_score(X_preprocessed, pred)

    print('Silhouette Score: %.3f' % score_kemans_s)
    print('Calinski Harabasz Score: %.3f' % score_kemans_c)
    print('Davies Bouldin Score: %.3f' % score_kemans_d)

    return score_kemans_s, score_kemans_c, score_kemans_d


def eval_all_metrics(data, preds, preds_enc, gt):
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, accuracy_score, precision_score, recall_score, f1_score
    for X, pred, pred_enc in zip(data, preds, preds_enc):
        sil = silhouette_score(X, pred, metric='euclidean')
        ch = calinski_harabasz_score(X, pred)
        db = davies_bouldin_score(X, pred)

        acc = accuracy_score(gt, pred_enc)
        rec = recall_score(gt, pred_enc, average='weighted')
        prec = precision_score(gt, pred_enc, average='weighted')
        f1 = f1_score(gt, pred_enc, average='weighted')

        print('Silhouette Score: %.3f' % sil)
        print('Calinski Harabasz Score: %.3f' % ch)
        print('Davies Bouldin Score: %.3f' % db)
        print('Accuracy Score: %.3f' % acc)
        print('Recall Score: %.3f' % rec)
        print('Precision Score: %.3f' % prec)
        print('F1 Score: %.3f' % f1)
        print('\n')
