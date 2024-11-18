import torch
import numpy as np
from torch import Tensor
import warnings
warnings.filterwarnings("ignore")

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def cos_similar1(p: Tensor, q: Tensor):
    print(p.shape, q.shape, type(p), type(q), q.transpose(0, -1).shape)
    sim_matrix = p.matmul(q.transpose(0, -1))
    a = torch.norm(p, dim=-1)
    b = torch.norm(q, dim=-1)
    # print("a , b : ", a.shape, b.shape)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return np.array(sim_matrix)

def cos_similar2(p: Tensor, q: Tensor):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, dim=-1)
    b = torch.norm(q, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix


def is_same(target, list):
    relation = []
    for i in range(len(list)):
        temp = 0
        for j in range(len(target)):
            if target[j] == 1 and list[i][j] == 1:
                temp = temp + 1
        relation.append(temp)
    return relation


def cos_similar_numpy(p, q):
    sim_matrix = np.zeros(len(q))
    for index in range(len(q)):
        sim_matrix[index] = float(np.dot(p, q[index])) / (np.linalg.norm(p) * np.linalg.norm(q[index]))
    return sim_matrix


def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    index_list = []

    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        d = np.dot(im, captions.T).flatten()
        inds = np.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = np.array([images[i] for i in range(0, len(images), 5)])

    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        d = np.dot(queries, ims.T)
        inds = np.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = np.argsort(d[i])[::-1]
            ranks[5 * index + i] = np.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):     # scores_i2t [1000, 5000], scores_t2i [5000, 1000]
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    # tr20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    # tr50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    # ir20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    # ir50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'i2t_@1': tr1,
                   'i2t_@5': tr5,
                   'i2t_@10': tr10,
                #    'i2t_@20': tr20,
                #    'i2t_@50': tr50,
                   'i2t_mean': tr_mean,
                   't2i_@1': ir1,
                   't2i_@5': ir5,
                   't2i_@10': ir10,
                #    't2i_@20': ir20,
                #    't2i_@50': ir50,
                   't2i_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


@torch.no_grad()
def uni_ndcg_i2i(embed_1, embed_2, relmatrix, sims, npts=None, threshold=500):
    if npts is None:
        npts = embed_1.shape[0]

    ndcgs = np.zeros(npts)
    for index in range(npts):
        im = embed_1[index].reshape(1, embed_1.shape[1])
        d = sims[index]

        d[index] = 0
        relmatrix[index][index] = 0
        inds = np.argsort(d)[::-1]

        # compute NDCG
        inds_threshold = inds[0:threshold]
        if not np.all(relmatrix == 0):
            rel_threshold = relmatrix[index][inds_threshold]
            rel_order_threshold = np.sort(relmatrix[index])[::-1][0:threshold]
            dcg = 0.0
            idcg = 0.0
            for ind_t in range(threshold):
                dcg += rel_threshold[ind_t] / np.log2(ind_t + 2)
                idcg += rel_order_threshold[ind_t] / np.log2(ind_t + 2)
            if idcg > 0:
                ndcgs[index] = dcg / idcg
    # Compute metrics
    ndcgs = ndcgs.mean() * 100
    print("I2I_@", threshold, ndcgs)
    return ndcgs


@torch.no_grad()
def uni_ndcg_t2t(embed_1, embed_2, relmatrix, sims, npts=None, threshold=500):
    if npts is None:
        npts = embed_1.shape[0]

    ndcgs = np.zeros(npts)
    for index in range(npts):
        im = embed_1[index].reshape(1, embed_1.shape[1])
        d = sims[index]

        d[5 * index] = 0
        relmatrix[index][5 * index] = 0
        inds = np.argsort(d)[::-1]

        # compute NDCG
        inds_threshold = inds[0:threshold]
        if not np.all(relmatrix == 0):
            rel_threshold = relmatrix[index][inds_threshold]
            rel_order_threshold = np.sort(relmatrix[index])[::-1][0:threshold]
            dcg = 0.0
            idcg = 0.0
            for ind_t in range(threshold):
                dcg += rel_threshold[ind_t] / np.log2(ind_t + 2)
                idcg += rel_order_threshold[ind_t] / np.log2(ind_t + 2)
            if idcg > 0:
                ndcgs[index] = dcg / idcg
    # Compute metrics
    ndcgs = ndcgs.mean() * 100
    print("T2T_@", threshold, ndcgs)
    return ndcgs


@torch.no_grad()
def uni_cal_ndcg_image(img_embs, image_label, dataname):
    relmatrix = np.dot(image_label, image_label.T)
    relmatrix = np.load("embed/" + dataname + "-rougeL.npy")
    if dataname == "coco":
        relmatrix = relmatrix.reshape(25000, -1)[:len(img_embs) * 5, :len(img_embs)][::5, :]
    else:
        relmatrix = relmatrix.reshape(5000, -1)[:len(img_embs) * 5, :len(img_embs)][::5, :]

    print("image shape : ", img_embs.shape, image_label.shape)
    print("relmatrix shape : ", relmatrix.shape)
    sims = cos_similar1(torch.tensor(img_embs).view(img_embs.shape[0], -1),
                       torch.tensor(img_embs).view(img_embs.shape[0], -1))

    return {"I2I_@10": uni_ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=10),
            "I2I_@20": uni_ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=20),
            "I2I_@50": uni_ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=50)}


@torch.no_grad()
def uni_cal_ndcg_text(cap_embs, cap_label, dataname):
    relmatrix = np.dot(cap_label, cap_label.T)
    relmatrix = np.load("embed/" + dataname + "-rougeL.npy")
    if dataname == "coco":
        relmatrix = relmatrix.reshape(25000, -1)[:len(cap_embs), :len(cap_embs)//5]
    else:
        relmatrix = relmatrix.reshape(5000, -1)[:len(cap_embs), :len(cap_embs) // 5]
    relmatrix = relmatrix.T
    query = cap_embs[::5, :]
    print("cap shape : ", query.shape, cap_embs.shape, cap_label.shape)
    print("relmatrix shape : ", relmatrix.shape)
    sims = cos_similar1(torch.tensor(cap_embs).view(cap_embs.shape[0], -1),
                       torch.tensor(cap_embs).view(cap_embs.shape[0], -1))
    sims = sims[::5, :]
    print("sims shape : ", sims.shape)
    return {"T2T_@10": uni_ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=10),
            "T2T_@20": uni_ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=20),
            "T2T_@50": uni_ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=50)}


def mix_ndcg_i2i(embed_1, embed_2, relmatrix, sims, npts=None, threshold=500):
    if npts is None:
        npts = embed_1.shape[0]

    ndcgs = np.zeros(npts)
    for index in range(npts):
        d = sims[index]
        inds = np.argsort(d)[::-1]

        # compute NDCG
        inds_threshold = inds[0:threshold]
        if not np.all(relmatrix == 0):
            rel_threshold = relmatrix[index][inds_threshold]
            rel_order_threshold = np.sort(relmatrix[index])[::-1][0:threshold]
            dcg = 0.0
            idcg = 0.0
            for ind_t in range(threshold):
                dcg += rel_threshold[ind_t] / np.log2(ind_t + 2)
                idcg += rel_order_threshold[ind_t] / np.log2(ind_t + 2)
            if idcg > 0:
                ndcgs[index] = dcg / idcg

    # Compute metrics
    ndcgs = ndcgs.mean() * 100
    print("I2IT_@", threshold, ndcgs)
    return ndcgs


def mix_ndcg_t2t(embed_1, embed_2, relmatrix, sims, npts=None, threshold=500):
    if npts is None:
        npts = embed_1.shape[0]

    ndcgs = np.zeros(npts)
    for index in range(npts):
        im = embed_1[index].reshape(1, embed_1.shape[1])
        d = sims[index]
        inds = np.argsort(d)[::-1]

        # compute NDCG
        inds_threshold = inds[0:threshold]
        if not np.all(relmatrix == 0):
            rel_threshold = relmatrix[index][inds_threshold]
            rel_order_threshold = np.sort(relmatrix[index])[::-1][0:threshold]
            dcg = 0.0
            idcg = 0.0
            for ind_t in range(threshold):
                dcg += rel_threshold[ind_t] / np.log2(ind_t + 2)
                idcg += rel_order_threshold[ind_t] / np.log2(ind_t + 2)
            if idcg > 0:
                ndcgs[index] = dcg / idcg
    # Compute metrics
    ndcgs = ndcgs.mean() * 100
    print("T2IT_@", threshold, ndcgs)
    return ndcgs


def mix_cal_ndcg_image(img_embs, cap_embs, image_label, scores_i2t, dataname):
    relmatrix_text = np.load("embed/" + dataname + "-rougeL.npy")
    if dataname == "coco":
        relmatrix_text = relmatrix_text.reshape(25000,-1)[:len(cap_embs), :len(cap_embs) // 5].T
    else:
        relmatrix_text = relmatrix_text.reshape(5000,-1)[:len(cap_embs), :len(cap_embs) // 5].T
    query = cap_embs[::5, :]
    print("cap shape : ", query.shape, cap_embs.shape)
    print("relmatrix_text shape : ", relmatrix_text.shape)
    # sims_text = cos_similar2(torch.from_numpy(query), torch.from_numpy(cap_embs)).numpy()
    sims_text = scores_i2t
    print('sims_text', sims_text.shape)

    relmatrix_image = np.load("embed/" + dataname + "-rougeL.npy")
    if dataname == "coco":
        relmatrix_image = relmatrix_image.reshape(25000,-1)[:len(img_embs) * 5, :len(img_embs)][::5, :]
    else:
        relmatrix_image = relmatrix_image.reshape(5000,-1)[:len(img_embs) * 5, :len(img_embs)][::5, :]
    sims_image = cos_similar2(torch.from_numpy(img_embs), torch.from_numpy(img_embs)).numpy()
    print("relmatrix_image shape : ", relmatrix_image.shape)
    print('sims_image', sims_image.shape, sims_text.shape)
    relmatrix = np.concatenate((relmatrix_image, relmatrix_text), axis=1)
    sims = np.concatenate((sims_image, sims_text), axis=1)
    print('relmatrix', relmatrix.shape)
    print('sims', sims.shape)

    return {
            # "I2IT_@1": mix_ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=1),
            # "I2IT_@5": mix_ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=5),
            "I2IT_@10": mix_ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=10),
            "I2IT_@20": mix_ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=20),
            "I2IT_@50": mix_ndcg_i2i(img_embs, img_embs, relmatrix, sims, npts=None, threshold=50),
            }


def mix_cal_ndcg_text(cap_embs, img_embs, cap_label, scores_t2i, dataname):
    relmatrix_text = np.load("embed/" + dataname + "-rougeL.npy")
    if dataname == "coco":
        relmatrix_text = relmatrix_text.reshape(25000,-1)[:len(cap_embs), :len(cap_embs) // 5]
    else:
        relmatrix_text = relmatrix_text.reshape(5000,-1)[:len(cap_embs), :len(cap_embs) // 5]
    relmatrix_text = relmatrix_text.T
    query = cap_embs[::5, :]
    print("cap shape : ", query.shape, cap_embs.shape, cap_label.shape)
    print("relmatrix shape : ", relmatrix_text.shape)
    sims_text = cos_similar2(torch.from_numpy(query), torch.from_numpy(cap_embs)).numpy()
    print('sims_text', sims_text.shape)

    relmatrix_image = np.load("embed/" + dataname + "-rougeL.npy")
    if dataname == "coco":
        relmatrix_image = relmatrix_image.reshape(25000,-1)[:len(img_embs) * 5, :len(img_embs)][::5, :].T
    else:
        relmatrix_image = relmatrix_image.reshape(5000,-1)[:len(img_embs) * 5, :len(img_embs)][::5, :].T
    sims_image = cos_similar2(torch.from_numpy(img_embs), torch.from_numpy(img_embs)).numpy()
    sims_image = scores_t2i.T[:, ::5]
    print("relmatrix_image shape : ", relmatrix_image.shape)
    print('sims_image', sims_image.shape)

    relmatrix = np.concatenate((relmatrix_text, relmatrix_image), axis=1)
    sims = np.concatenate((sims_text, sims_image), axis=1)
    print('relmatrix', relmatrix.shape)
    print('sims', sims.shape)

    return {
            # "T2IT_@1": mix_ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=1),
            # "T2IT_@5": mix_ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=5),
            "T2IT_@10": mix_ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=10),
            "T2IT_@20": mix_ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=20),
            "T2IT_@50": mix_ndcg_t2t(query, cap_embs, relmatrix, sims, npts=None, threshold=50),
            }

# scores_i2t [1000, 5000], scores_t2i [5000, 1000], image_feat [1000, 145, 1024], image_embed [1000, 256], text_feat[5000, 60, 768], text_embed[5000, 256]
def get_score(scores_i2t, scores_t2i, image_feat, image_embed, text_feat, text_embed, txt2img, img2txt, dataname, op):  # txt2img [5000], img2txt [1000, 5]
    result = {}
    
    # cross-modal
    print("-----------------")
    result.update(itm_eval(scores_i2t, scores_t2i, txt2img, img2txt))
    print(result)

    # 1024 768
    if op == 1024:
        image_embed = image_feat[:, 0, :]
        text_embed = text_feat[:, 0, :]
    
    # uni-modal
    print("-----------------------------------------------")
    result.update(uni_cal_ndcg_image(image_embed, image_embed, dataname))
    result.update(uni_cal_ndcg_text(text_embed, text_embed, dataname))

    # mixed-retrieval
    print("-----------------------------------------------")
    result.update(mix_cal_ndcg_image(image_embed, text_embed, image_embed, scores_i2t, dataname))
    result.update(mix_cal_ndcg_text(text_embed, image_embed, text_embed, scores_t2i, dataname))
    return result
