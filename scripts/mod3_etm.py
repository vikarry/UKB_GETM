import torch
import torch.nn.functional as F
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ETM(nn.Module):
    def __init__(self, num_topics, vocab_size1, vocab_size2, vocab_size3, t_hidden_size, rho_size,
                 theta_act, predcoef, embeddings1=None, embeddings2=None, embeddings3=None,
                 train_embeddings1=True, train_embeddings2=True, train_embeddings3=True,
                 rho_fixed1=False, rho_fixed2=False, rho_fixed3=False, enc_drop=0.5):
        '''
        Create a GETM with three modalities
        :param num_topics: The number of topics
        :param vocab_size1: the size of first vocabulary (e.g., medications)
        :param vocab_size2: the size of second vocabulary (e.g., conditions)
        :param vocab_size3: the size of third vocabulary (e.g., procedures)
        :param t_hidden_size: the size of hidden layer size
        :param rho_size: the dimension of embeddings
        :param theta_act: activation function of the encoder
        :param predcoef: coefficient to balance loss terms scale
        :param embeddings1: first modality pre-trained embedding
        :param embeddings2: second modality pre-trained embedding
        :param embeddings3: third modality pre-trained embedding
        :param train_embeddings1: whether to use first modality pre-trained embedding
        :param train_embeddings2: whether to use second modality pre-trained embedding
        :param train_embeddings3: whether to use third modality pre-trained embedding
        :param rho_fixed1: whether to fix first modality embedding during training
        :param rho_fixed2: whether to fix second modality embedding during training
        :param rho_fixed3: whether to fix third modality embedding during training
        :param enc_drop: dropout rate
        '''
        super(ETM, self).__init__()

        ## define hyperparameters
        self.num_topics = num_topics
        self.vocab_size1 = vocab_size1
        self.vocab_size2 = vocab_size2
        self.vocab_size3 = vocab_size3
        self.t_hidden_size = t_hidden_size
        self.rho_size = rho_size
        self.enc_drop = enc_drop
        self.t_drop = nn.Dropout(enc_drop)
        self.predcoef = predcoef

        self.theta_act = self.get_activation(theta_act)

        self.train_embeddings1 = train_embeddings1
        self.rho_fixed1 = rho_fixed1

        self.train_embeddings2 = train_embeddings2
        self.rho_fixed2 = rho_fixed2

        self.train_embeddings3 = train_embeddings3
        self.rho_fixed3 = rho_fixed3

        ## define the word embedding matrices \rho for each modality
        if self.train_embeddings1:
            self.rho1 = nn.Linear(rho_size, vocab_size1, bias=True)  # L x V
        else:
            if not rho_fixed1:
                num_embeddings, emsize = embeddings1.size()
                self.rho1 = nn.Embedding(num_embeddings, emsize)
                self.rho1.weight.data.copy_(embeddings1)
                self.rho1.weight.requires_grad = True
            else:
                self.rho1 = embeddings1

        if self.train_embeddings2:
            self.rho2 = nn.Linear(rho_size, vocab_size2, bias=True)  # L x V
        else:
            if not rho_fixed2:
                num_embeddings, emsize = embeddings2.size()
                self.rho2 = nn.Embedding(num_embeddings, emsize)
                self.rho2.weight.data.copy_(embeddings2)
                self.rho2.weight.requires_grad = True
            else:
                self.rho2 = embeddings2

        if self.train_embeddings3:
            self.rho3 = nn.Linear(rho_size, vocab_size3, bias=True)  # L x V
        else:
            if not rho_fixed3:
                num_embeddings, emsize = embeddings3.size()
                self.rho3 = nn.Embedding(num_embeddings, emsize)
                self.rho3.weight.data.copy_(embeddings3)
                self.rho3.weight.requires_grad = True
            else:
                self.rho3 = embeddings3

        ## define the matrices containing the topic embeddings for each modality
        self.alphas1 = nn.Linear(rho_size, num_topics, bias=False)
        self.alphas2 = nn.Linear(rho_size, num_topics, bias=False)
        self.alphas3 = nn.Linear(rho_size, num_topics, bias=False)

        ## define variational distribution for \theta_{1:D} via amortization
        self.q_theta = nn.Sequential(
            nn.Linear(vocab_size1 + vocab_size2 + vocab_size3, t_hidden_size),
            self.theta_act,
            nn.BatchNorm1d(t_hidden_size),
            nn.Linear(t_hidden_size, t_hidden_size),
            self.theta_act,
            nn.BatchNorm1d(t_hidden_size),
        )
        self.mu_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(t_hidden_size, num_topics, bias=True)

    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def encode(self, bows):
        """Returns parameters of the variational distribution for \theta.

        input: bows
                batch of bag-of-words...tensor of shape bsz x V
        output: mu_theta, log_sigma_theta
        """
        q_theta = self.q_theta(bows)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)

        # KL[q(theta)||p(theta)] = lnq(theta) - lnp(theta)
        kl_theta = -0.5 * torch.sum(1 + logsigma_theta - mu_theta.pow(2) - logsigma_theta.exp(), dim=-1).mean()
        return mu_theta, logsigma_theta, kl_theta

    def calc_beta(self, rho, alphas):
        '''
        Calculate topic mixture
        :param rho: feature embedding
        :param alphas: topic embedding
        :return: beta
        '''
        try:
            logit = alphas(rho.weight)
        except:
            logit = alphas(rho)
        beta = F.softmax(logit, dim=0).transpose(1, 0)  ## softmax over vocab dimension
        return beta

    def get_beta(self):
        # modality 1-specific topic mixture
        beta1 = self.calc_beta(self.rho1, self.alphas1)
        # modality 2-specific topic mixture
        beta2 = self.calc_beta(self.rho2, self.alphas2)
        # modality 3-specific topic mixture
        beta3 = self.calc_beta(self.rho3, self.alphas3)
        return beta1, beta2, beta3

    def get_theta(self, normalized_bows):
        mu_theta, logsigma_theta, kld_theta = self.encode(normalized_bows)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        return theta, kld_theta, mu_theta, logsigma_theta

    def decode(self, theta, beta):
        res = torch.mm(theta, beta)
        preds = torch.log(res + 1e-6)
        return preds

    def forward(self, bows, normalized_bows, y_true=None, mask=None, weights=None, theta=None, aggregate=True):
        ## get \theta
        if theta is None:
            theta, kld_theta, x, _ = self.get_theta(normalized_bows)
        else:
            kld_theta = None

        # Split bows into the three modalities
        bows1 = bows[:, :self.vocab_size1]
        bows2 = bows[:, self.vocab_size1:self.vocab_size1 + self.vocab_size2]
        bows3 = bows[:, self.vocab_size1 + self.vocab_size2:]

        ## get \beta for all three modalities
        beta1, beta2, beta3 = self.get_beta()

        ## get prediction loss for all three modalities
        preds1 = self.decode(theta, beta1)
        preds2 = self.decode(theta, beta2)
        preds3 = self.decode(theta, beta3)

        recon_loss = -(preds1 * bows1).sum(1) - (preds2 * bows2).sum(1) - (preds3 * bows3).sum(1)

        if aggregate:
            recon_loss = recon_loss.mean()

        return recon_loss * self.predcoef, kld_theta