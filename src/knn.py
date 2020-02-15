import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pdb; pdb.set_trace()  # XXX BREAKPOINT
import torch
import utils as u
from torchvision import datasets, transforms
from nca import NCA
device=torch.device("cuda")
transform = transforms.Compose([
    transforms.ToTensor(),
])
batch=100
train = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                        None, validation=False, by_subject = True, batch_size=batch,
                        scale = True, rectify=False, sample_0=False, step=1)
test = u.NinaGenerator("../data/ninaPro", ['b'], [u.butter_highpass_filter],
                       None, validation=True, by_subject = True, batch_size=batch,
                       scale = True, rectify = False, sample_0=False, step=1)
outs = [[] for i in range(4)]
for tr, te in zip(train, test):
    outs[0].append(tr[0].reshape((tr[0].shape[0], -1),order='f'))
    outs[1].append(tr[1])
    outs[2].append(te[0].reshape((tr[0].shape[0], -1), order='f'))
    outs[3].append(te[1])

x_tr,  x_te = (torch.FloatTensor(np.concatenate(x,0)) for x in outs[0::2])
y_tr,  y_te = (torch.LongTensor(np.concatenate(x,0)) for x in outs[1::2])


nca = NCA(dim=52, max_iters=1000, tol=1e-9)
nca.train(x_tr, y_tr, batch_size=10000, weight_decay=10, lr=1e-5, normalize=False)

A = nca.A.detach().cpu().numpy()
x_tr = x_tr.cpu().numpy()
y_tr = y_tr.cpu().numpy()
x_te = x_te.cpu().numpy()
y_te = y_te.cpu().numpy()

x_tr_emb = x_tr @ A.T
x_te_emb = x_te @ A.T

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(x_tr_emb, y_tr)
pred = knn.predict(x_te_emb)


print("nca_acc: {}".format(accuracy_score(pred, y_te)))


knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(x_tr, y_tr)
pred = knn.predict(x_te)
print("normal_acc: {}".format(accuracy_score(pred, y_te)))
import pdb; pdb.set_trace()  # XXX BREAKPOINT

