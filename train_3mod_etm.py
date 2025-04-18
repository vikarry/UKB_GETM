from __future__ import print_function

import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from torch import optim
from scripts.mod3_etm import ETM
from scripts.utils import get_topic_coherence, get_topic_diversity
from scripts.dataset import PatientDrugDataset
import pickle
import math

parser = argparse.ArgumentParser(description='The Embedded Topic Model with 3 Modalities')

parser.add_argument('--data_path', type=str, default='input_data', help='directory containing data')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to load data')

parser.add_argument('--save_path', type=str, default='results', help='path to save results')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training')

### model-related arguments
parser.add_argument('--vocab_size1', type=int, default=802, help='number of unique drugs')
parser.add_argument('--vocab_size2', type=int, default=443, help='number of unique conditions')
parser.add_argument('--vocab_size3', type=int, default=10, help='number of unique third modality items')
parser.add_argument('--num_topics', type=int, default=128, help='number of topics')
parser.add_argument('--rho_size', type=int, default=128, help='dimension of rho')
parser.add_argument('--t_hidden_size', type=int, default=64, help='dimension of hidden space of q(theta)')
parser.add_argument('--theta_act', type=str, default='relu',
                    help='tanh, softplus, relu, rrelu, leakyrelu, elu, selu, glu)')
parser.add_argument('--predcoef', type=int, default=2, help='coefficient for kl loss')

parser.add_argument('--train_embeddings1', type=int, default=0, help='whether to include pretrained embedding')
parser.add_argument('--rho_fixed1', type=int, default=1, help='whether to fix rho or train it')
parser.add_argument('--train_embeddings2', type=int, default=0, help='whether to include pretrained embedding')
parser.add_argument('--rho_fixed2', type=int, default=1, help='whether to fix rho or train it')
parser.add_argument('--train_embeddings3', type=int, default=0, help='whether to include pretrained embedding')
parser.add_argument('--rho_fixed3', type=int, default=1, help='whether to fix rho or train it')

parser.add_argument('--embedding1', type=str, default="vertex_embeddings.npy",
                    help='file contained pretrained rho for modality 1')
parser.add_argument('--embedding2', type=str, default="vertex_embeddings.npy",
                    help='file contained pretrained rho for modality 2')
parser.add_argument('--embedding3', type=str, default="vertex_embeddings.npy",
                    help='file contained pretrained rho for modality 3')
parser.add_argument('--label_name', type=str, default="y_age", help='file contained age label')
parser.add_argument('--mask_name', type=str, default="mask_age", help='file contained mask for age label')
parser.add_argument('--X_name', type=str, default="bow", help='file contained input for age label')
parser.add_argument('--Xt_name', type=str, default="bow_t", help='file contained time-varying input for age label')
### optimization-related arguments
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_factor', type=float, default=4.0, help='divide learning rate by this...')

parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train...150 for 20ng 100 for others')

parser.add_argument('--mode', type=str, default='train', help='train or eval model')
parser.add_argument('--optimizer', type=str, default='adam', help='choice of optimizer')
parser.add_argument('--seed', type=int, default=2020, help='random seed (default: 1)')

parser.add_argument('--enc_drop', type=float, default=0.1, help='dropout rate on encoder')
parser.add_argument('--lstm_dropout', type=float, default=0.0, help='dropout rate on rnn for prediction')
parser.add_argument('--clip', type=float, default=2.0, help='gradient clipping')

parser.add_argument('--nonmono', type=int, default=10, help='number of bad hits allowed')
parser.add_argument('--wdecay', type=float, default=1.2e-6, help='some l2 regularization')

parser.add_argument('--anneal_lr', type=int, default=1, help='whether to anneal the learning rate or not')
parser.add_argument('--bow_norm', type=int, default=1, help='normalize the bows or not')

### evaluation, visualization, and logging-related arguments
parser.add_argument('--num_words', type=int, default=20, help='number of words for topic viz')
parser.add_argument('--log_interval', type=int, default=1, help='when to log training')

parser.add_argument('--visualize_every', type=int, default=1, help='when to visualize results')
parser.add_argument('--eval_batch_size', type=int, default=1000, help='input batch size for evaluation')
parser.add_argument('--load_from', type=str, default='', help='the name of the ckpt to eval from')
parser.add_argument('--tc', type=int, default=0, help='whether to compute topic coherence or not')
parser.add_argument('--td', type=int, default=0, help='whether to compute topic diversity or not')
parser.add_argument('--gpu_device', type=str, default="cuda", help='gpu device name')

args = parser.parse_args()

device = torch.device(args.gpu_device if torch.cuda.is_available() else "cpu")

print('\n')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

embeddings1 = None
if not args.train_embeddings1:
    embed_file1 = os.path.join(args.data_path, args.embedding1)
    embeddings1 = np.load(embed_file1)
    embeddings1 = torch.from_numpy(embeddings1).float().to(device)

embeddings2 = None
if not args.train_embeddings2:
    embed_file2 = os.path.join(args.data_path, args.embedding2)
    embeddings2 = np.load(embed_file2)
    embeddings2 = torch.from_numpy(embeddings2).float().to(device)

embeddings3 = None
if not args.train_embeddings3:
    embed_file3 = os.path.join(args.data_path, args.embedding3)
    embeddings3 = np.load(embed_file3)
    embeddings3 = torch.from_numpy(embeddings3).float().to(device)

## define checkpoint
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if args.mode == 'eval':
    ckpt = args.load_from
else:
    ckpt = os.path.join(args.save_path,
                        'etm_UKPD_K_{}_Htheta_{} Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
                            args.num_topics, args.t_hidden_size, args.optimizer, args.clip, args.theta_act,
                            args.lr, args.batch_size, args.rho_size, args.train_embeddings1))

model = ETM(args.num_topics, args.vocab_size1, args.vocab_size2, args.vocab_size3,
            args.t_hidden_size, args.rho_size, args.theta_act, args.predcoef,
            embeddings1, embeddings2, embeddings3,
            args.train_embeddings1, args.train_embeddings2, args.train_embeddings3,
            args.rho_fixed1, args.rho_fixed2, args.rho_fixed3,
            args.enc_drop).to(device)

print('model: {}'.format(model))

if args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'adadelta':
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'rmsprop':
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
elif args.optimizer == 'asgd':
    optimizer = optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
else:
    print('Defaulting to vanilla SGD')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)


def train(epoch):
    model.train()
    acc_loss = 0
    acc_kl_theta_loss = 0
    cnt = 0

    train_t_filename = os.path.join(args.data_path, f"{args.X_name}_train_cmskp_3mod.npy")

    y_filename = None
    mask_filename = None

    TrainDataset = PatientDrugDataset(train_t_filename, y_filename, mask_filename)
    TrainDataloader = DataLoader(TrainDataset, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.num_workers)

    for idx, (sample_batch, index) in enumerate(TrainDataloader):
        optimizer.zero_grad()
        model.zero_grad()
        data_batch = sample_batch['Data'].float().to(device)
        # data_batch = torch.transpose(data_batch_t, 0, 1).reshape(data_batch_t.size(0)*data_batch_t.size(1), data_batch_t.size(2))

        sums = data_batch.sum(1).unsqueeze(1)
        if args.bow_norm:
            normalized_data_batch = data_batch / sums
        else:
            normalized_data_batch = data_batch

        recon_loss, kld_theta = model(data_batch, normalized_data_batch)

        # NELBO = -(loglikelihood - KL[q||p]) = -loglikelihood + KL[q||p]
        total_loss = recon_loss + kld_theta

        total_loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        acc_loss += torch.sum(recon_loss).item()
        acc_kl_theta_loss += torch.sum(kld_theta).item()

        cnt += 1

        if idx % args.log_interval == 0 and idx > 0:
            cur_loss = round(acc_loss / cnt, 2)
            cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)

            cur_real_loss = round(cur_loss + cur_kl_theta, 2)

            # print('Epoch: {} .. batch: {}/320 .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
            #     epoch, idx, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))

    cur_loss = round(acc_loss / cnt, 2)
    cur_kl_theta = round(acc_kl_theta_loss / cnt, 2)

    cur_real_loss = round(cur_loss + cur_kl_theta, 2)
    print('*' * 100)
    print('Epoch----->{} .. LR: {} .. KL_theta: {} .. Rec_loss: {} .. NELBO: {}'.format(
        epoch, optimizer.param_groups[0]['lr'], cur_kl_theta, cur_loss, cur_real_loss))
    print('*' * 100)


def evaluate(m, tc=False, td=False):
    """Compute perplexity on document completion.
    """
    m.eval()
    with torch.no_grad():
        test_t_filename = os.path.join(args.data_path, f"{args.X_name}_test_cmskp_3mod.npy")

        y_filename = None
        mask_filename = None

        TestDataset = PatientDrugDataset(test_t_filename, y_filename, mask_filename)
        TestDataloader = DataLoader(TestDataset, batch_size=args.eval_batch_size,
                                    shuffle=True, num_workers=args.num_workers)
        beta1, beta2, beta3 = m.get_beta()
        acc_loss = 0
        cnt = 0
        for idx, (sample_batch, index) in enumerate(TestDataloader):
            ### do dc and tc here
            ## get theta from first half of docs
            data_batch = sample_batch['Data'].float().to(device)
            # data_batch = torch.transpose(data_batch_t, 0, 1).reshape(data_batch_t.size(0) * data_batch_t.size(1),
            #                                                          data_batch_t.size(2))
            sums = data_batch.sum(1).unsqueeze(1)
            if args.bow_norm:
                normalized_data_batch = data_batch / sums
            else:
                normalized_data_batch = data_batch
            theta, _, x, _ = m.get_theta(normalized_data_batch)

            res1 = torch.mm(theta, beta1)
            preds1 = torch.log(res1)
            res2 = torch.mm(theta, beta2)
            preds2 = torch.log(res2)
            res3 = torch.mm(theta, beta3)
            preds3 = torch.log(res3)

            bows1 = data_batch[:, :args.vocab_size1]
            bows2 = data_batch[:, args.vocab_size1:args.vocab_size1 + args.vocab_size2]
            bows3 = data_batch[:, args.vocab_size1 + args.vocab_size2:]

            recon_loss = -(preds1 * bows1).sum(1) - (preds2 * bows2).sum(1) - (preds3 * bows3).sum(1)

            loss = recon_loss.mean().item()

            acc_loss += loss
            cnt += 1
        print('*' * 100)
        print(cnt * normalized_data_batch.shape[0])
        cur_loss = acc_loss / cnt
        ppl_dc = round(math.exp(cur_loss), 1)
        print("Perplexity: {:.4f}".format(ppl_dc))

        print('*' * 100)
        print('Test Loss: {}'.format(cur_loss))
        print('*' * 100)

        TQ = TC = TD = 0

        if tc or td:
            beta = theta.data.cpu().numpy()

        if tc:
            print('Computing topic coherence...')
            TC_all, cnt_all = get_topic_coherence(beta, train_tokens, vocab)

            TC_all = torch.tensor(TC_all)
            cnt_all = torch.tensor(cnt_all)
            TC_all = TC_all / cnt_all
            TC_all[TC_all < 0] = 0

            TC = TC_all.mean().item()
            print('Topic Coherence is: ', TC)
            print('\n')

        if td:
            print('Computing topic diversity...')
            TD_all = get_topic_diversity(beta, 25)
            TD = np.mean(TD_all)
            print('Topic Diversity is: {}'.format(TD))

            print('Get topic quality...')
            TQ = TD * TC
            print('Topic Quality is: {}'.format(TQ))
            print('#' * 100)

        return ppl_dc, TQ, TC, TD


def load_model_and_generate_embeddings(args, input_file=None):
    """
    Load a pre-trained ETM model and generate theta, mu_theta for all patients in the input file.

    Args:
        args: Existing argument object from the main script
        input_file (str, optional): Path to the input file containing patient data.
                                   If None, uses the file specified in args.
    """
    # Set device
    device = torch.device(args.gpu_device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model from checkpoint
    ckpt = args.load_from if args.load_from else os.path.join(args.save_path,
                                                              'etm_UKPD_K_{}_Htheta_{} Optim_{}_Clip_{}_ThetaAct_{}_Lr_{}_Bsz_{}_RhoSize_{}_trainEmbeddings_{}'.format(
                                                                  args.num_topics, args.t_hidden_size, args.optimizer,
                                                                  args.clip, args.theta_act,
                                                                  args.lr, args.batch_size, args.rho_size,
                                                                  args.train_embeddings1))

    print(f"Loading model from {ckpt}")
    with open(ckpt, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()

    # Create output directory if it doesn't exist
    output_dir = os.path.join(args.save_path, "patient_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    # Determine which file to process
    if input_file is None:
        input_file = os.path.join(args.data_path, f"{args.X_name}_all_cmskp_3mod.npy")

    print(f"Processing data from {input_file}")
    dataset = PatientDrugDataset(input_file)

    # Process all patients at once or in a single batch to avoid splitting into multiple files
    batch_size = len(dataset)  # Process all at once
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # Get the single batch of all patients
    for _, (sample_batch, index) in enumerate(dataloader):
        # Move data to device
        data_batch = sample_batch['Data'].float().to(device)

        # Generate embeddings
        with torch.no_grad():
            theta, _, mu_theta, logsigma_theta = model.get_theta(data_batch)

        # Convert to numpy
        theta_np = theta.detach().cpu().numpy()
        mu_theta_np = mu_theta.detach().cpu().numpy()
        logsigma_theta_np = logsigma_theta.detach().cpu().numpy()

        # Save the embeddings as single files
        saved_theta = os.path.join(output_dir, "theta.npy")
        saved_mu = os.path.join(output_dir, "mu_theta.npy")
        saved_log = os.path.join(output_dir, "log_theta.npy")

        np.save(saved_theta, theta_np)
        np.save(saved_mu, mu_theta_np)
        np.save(saved_log, logsigma_theta_np)

        # Save indices
        saved_index = os.path.join(output_dir, "indices.pkl")
        with open(saved_index, "wb") as f:
            pickle.dump(index.cpu().numpy(), f)

        print(f"Completed! Generated embeddings for {len(index)} patients.")
        print(f"Outputs saved to {output_dir}")

    return output_dir


def add_embedding_generation(args):
    """
    Function to add to the end of the main script to generate embeddings for all patients.
    """
    print("\n" + "=" * 50)
    print("Generating embeddings for all patients...")

    # For train data
    train_file = os.path.join(args.data_path, f"{args.X_name}_train_cmskp_3mod.npy")
    train_output = load_model_and_generate_embeddings(args, train_file)
    print(f"Train embeddings saved to {train_output}")

    # For test data
    test_file = os.path.join(args.data_path, f"{args.X_name}_test_cmskp_3mod.npy")
    test_output = load_model_and_generate_embeddings(args, test_file)
    print(f"Test embeddings saved to {test_output}")

    # If there's a combined all file, process that too
    all_file = os.path.join(args.data_path, f"{args.X_name}_all_cmskp_3mod.npy")
    if os.path.exists(all_file):
        all_output = load_model_and_generate_embeddings(args, all_file)
        print(f"All embeddings saved to {all_output}")

    print("=" * 50)


if args.mode == 'train':
    ## train model on data
    best_epoch = 0
    best_val_ppl = 1e100
    all_val_ppls = []

    for epoch in range(1, args.epochs):
        train(epoch)
        val_ppl, tq, tc, td = evaluate(model)
        if val_ppl < best_val_ppl:
            with open(ckpt, 'wb') as f:
                torch.save(model, f)
            best_epoch = epoch
            best_val_ppl = val_ppl
        else:
            ## check whether to anneal lr
            lr = optimizer.param_groups[0]['lr']
            if args.anneal_lr and (
                    len(all_val_ppls) > args.nonmono and val_ppl > min(all_val_ppls[:-args.nonmono]) and lr > 1e-5):
                optimizer.param_groups[0]['lr'] /= args.lr_factor

        all_val_ppls.append(val_ppl)

    with open(ckpt, 'rb') as f:
        model = torch.load(f, weights_only=False)
    model = model.to(device)
    val_ppl = evaluate(model)
else:
    with open(ckpt, 'rb') as f:
        model = torch.load(f, weights_only=False)
    model = model.to(device)

model.eval()

## show topics
beta1, beta2, beta3 = model.get_beta()
beta1 = beta1.detach().cpu().numpy()
beta2 = beta2.detach().cpu().numpy()
beta3 = beta3.detach().cpu().numpy()

saved_file = os.path.join(args.save_path, "beta1.npy")
np.save(saved_file, beta1)

saved_file = os.path.join(args.save_path, "beta2.npy")
np.save(saved_file, beta2)

saved_file = os.path.join(args.save_path, "beta3.npy")
np.save(saved_file, beta3)

saved_rho1 = os.path.join(args.save_path, "rho1.npy")
try:
    rho1 = model.rho1.weight.detach().cpu().numpy()
except:
    rho1 = model.rho1.detach().cpu().numpy()
np.save(saved_rho1, rho1)

saved_rho2 = os.path.join(args.save_path, "rho2.npy")
try:
    rho2 = model.rho2.weight.detach().cpu().numpy()
except:
    rho2 = model.rho2.detach().cpu().numpy()
np.save(saved_rho2, rho2)

saved_rho3 = os.path.join(args.save_path, "rho3.npy")
try:
    rho3 = model.rho3.weight.detach().cpu().numpy()
except:
    rho3 = model.rho3.detach().cpu().numpy()
np.save(saved_rho3, rho3)

##############################################################################
# Saved variables for future analysis
saved_alpha1 = os.path.join(args.save_path, "alpha1.npy")
alpha1 = model.alphas1.weight.detach().cpu().numpy()
np.save(saved_alpha1, alpha1)

saved_alpha2 = os.path.join(args.save_path, "alpha2.npy")
alpha2 = model.alphas2.weight.detach().cpu().numpy()
np.save(saved_alpha2, alpha2)

saved_alpha3 = os.path.join(args.save_path, "alpha3.npy")
alpha3 = model.alphas3.weight.detach().cpu().numpy()
np.save(saved_alpha3, alpha3)

filename_t = os.path.join(args.data_path, f"{args.X_name}_train_cmskp_3mod.npy")

Dataset = PatientDrugDataset(filename_t)
MyDataloader = DataLoader(Dataset, batch_size=1000,
                          shuffle=False, num_workers=args.num_workers)

index_list = []
for idx, (sample_batch, index) in enumerate(MyDataloader):
    index_list.append(index.cpu().numpy())
    data_batch = sample_batch['Data'].float().to(device)

    theta, _, mu_theta, logsigma_theta = model.get_theta(data_batch)

    theta = theta.detach().cpu().numpy()
    mu_theta = mu_theta.detach().cpu().numpy()
    logsigma_theta = logsigma_theta.detach().cpu().numpy()

    saved_folder = os.path.join(args.save_path, "theta_train")
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    saved_theta = os.path.join(saved_folder, f"theta{idx}.npy")
    saved_mu = os.path.join(saved_folder, f"mu_theta{idx}.npy")
    saved_log = os.path.join(saved_folder, f"log_theta{idx}.npy")

    np.save(saved_theta, theta)
    np.save(saved_mu, mu_theta)
    np.save(saved_log, logsigma_theta)

saved_index = os.path.join(saved_folder, "index.pkl")
with open(saved_index, "wb") as f:
    pickle.dump(index_list, f)

filename_t = os.path.join(args.data_path, f"{args.X_name}_test_cmskp_3mod.npy")

y_filename = None
mask_filename = None

Dataset = PatientDrugDataset(filename_t, y_filename, mask_filename)
MyDataloader = DataLoader(Dataset, batch_size=1000,
                          shuffle=False, num_workers=args.num_workers)

index_list = []
for idx, (sample_batch, index) in enumerate(MyDataloader):
    index_list.append(index.cpu().numpy())
    data_batch = sample_batch['Data'].float().to(device)

    theta, _, mu_theta, logsigma_theta = model.get_theta(data_batch)

    theta = theta.detach().cpu().numpy()
    mu_theta = mu_theta.detach().cpu().numpy()
    logsigma_theta = logsigma_theta.detach().cpu().numpy()

    saved_folder = os.path.join(args.save_path, "theta_test")
    if not os.path.exists(saved_folder):
        os.makedirs(saved_folder)
    saved_theta = os.path.join(saved_folder, f"theta{idx}.npy")
    saved_mu = os.path.join(saved_folder, f"mu_theta{idx}.npy")
    saved_log = os.path.join(saved_folder, f"log_theta{idx}.npy")

    np.save(saved_theta, theta)
    np.save(saved_mu, mu_theta)
    np.save(saved_log, logsigma_theta)

saved_index = os.path.join(saved_folder, "index.pkl")
with open(saved_index, "wb") as f:
    pickle.dump(index_list, f)

# evaluate(model)
# add_embedding_generation(args)