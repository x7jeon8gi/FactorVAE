# GRU을 통한 feature extraction, 입력으로 주식들의 firm characteristic을 받아서, firm characteristic을 통해 주식의 latent vector를 추출
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np

class FeatureExtractor(nn.Module):
    def __init__(self, num_latent, hidden_size, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.num_latent = num_latent
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.normalize = nn.LayerNorm(num_latent)
        self.linear = nn.Linear(num_latent, num_latent)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(num_latent, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        #! x: (batch_size, seq_length, num_latent)
        # Apply linear and LeakyReLU activation
        #* layer norm 추가
        x = self.normalize(x)
        out = self.linear(x)
        out = self.leakyrelu(out)
        # Forward propagate GRU
        stock_latent, _ = self.gru(out)
        return stock_latent[:,-1,:] #* stock_latent[-1]: (batch_size, hidden_size)

class FactorEncoder(nn.Module):
    def __init__(self, num_factors, num_portfolio, hidden_size):
        super(FactorEncoder, self).__init__()
        self.num_factors = num_factors
        self.linear = nn.Linear(hidden_size, num_portfolio)
        self.softmax = nn.Softmax(dim=1)
        
        self.linear2 = nn.Linear(num_portfolio, num_factors)
        self.softplus = nn.Softplus()
        
    def mapping_layer(self, portfolio_return):
        #! portfolio_return: (batch_size, 1)
        #! mapping layer
        # print(portfolio_return.shape)
        mean = self.linear2(portfolio_return.squeeze(1))
        sigma = self.softplus(mean)
        return mean, sigma
    
    def forward(self, stock_latent, returns):
        #! stock_latent: (batch_size, hidden_size)
        #! returns: (batch_size, 1) (딱 한 기간의 수익률)
        #! make portfolio
        weights = self.linear(stock_latent)
        weights = self.softmax(weights) # (batch_size, num_portfolio)

        # multiply weights and returns
        #print(f"weights shape: {weights.shape}, returns shape: {returns.shape}") # [300, 20], [300, 1]
        # check returns.shape is tuple
        if returns.dim() == 1:
            returns = returns.unsqueeze(1)
        portfolio_return = torch.mm(weights.transpose(1,0), returns) #* portfolio_return: (M, 1)
        #print(f"portfolio_return shape: {portfolio_return.shape}")
        
        return self.mapping_layer(portfolio_return)

class AlphaLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AlphaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()
        
    def forward(self, stock_latent):
        #* stock latent는 FeatureExtractor에서 나온 것 (batch_size, hidden_size)
        stock_latent = self.linear1(stock_latent)
        stock_latent = self.leakyrelu(stock_latent)
        alpha_mu = self.mu_layer(stock_latent)
        alpha_sigma = self.sigma_layer(stock_latent)
        return alpha_mu, self.softplus(alpha_sigma)
        
class BetaLayer(nn.Module):
    """calcuate factor exposure beta(N*K)"""
    def __init__(self, hidden_size, num_factors):
        super(BetaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, num_factors)
    
    def forward(self, stock_latent):
        beta = self.linear1(stock_latent)
        return beta
        
class FactorDecoder(nn.Module):
    def __init__(self, alpha_layer, beta_layer):
        super(FactorDecoder, self).__init__()

        self.alpha_layer = alpha_layer
        self.beta_layer = beta_layer
    
    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma)
        return mu + eps * sigma
    
    def forward(self, stock_latent, factor_mu, factor_sigma):
        #! warning: alpha_mu, alpha_sigma -> (N), (N)
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        #print(f"alpha_mu shape: {alpha_mu.shape}, alpha_sigma shape: {alpha_sigma.shape}")
        beta = self.beta_layer(stock_latent)

        factor_mu = factor_mu.view(-1, 1)
        factor_sigma = factor_sigma.view(-1, 1)

        # Replace any zero values in factor_sigma with a small value
        factor_sigma[factor_sigma == 0] = 1e-6
        #print(f"factor_mu shape: {factor_mu.shape}, factor_sigma shape: {factor_sigma.shape}")
        #print(f"beta shape: {beta.shape}")
        mu = alpha_mu + torch.matmul(beta, factor_mu)
        sigma = torch.sqrt(alpha_sigma**2 + torch.matmul(beta**2, factor_sigma**2) + 1e-6)

        return self.reparameterize(mu, sigma)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        
        self.query = nn.Parameter(torch.randn(hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, stock_latent):
        #* calculate attention weights

        self.key = self.key_layer(stock_latent)
        self.value = self.value_layer(stock_latent)
        
        attention_weights = torch.matmul(self.query, self.key.transpose(1,0)) # (N)
        #* scaling
        attention_weights = attention_weights / torch.sqrt(torch.tensor(self.key.shape[0])+ 1e-6)
        # print(f"attention_weights shape: {attention_weights.shape}")
        attention_weights = self.dropout(attention_weights)
        attention_weights = F.relu(attention_weights) # max(0, x)
        attention_weights = F.softmax(attention_weights, dim=0) # (N)
        
        #! calculate context vector
        if torch.isnan(attention_weights).any() or torch.isinf(attention_weights).any():
            return torch.zeros_like(self.value[0])
        else:
            context_vector = torch.matmul(attention_weights, self.value) # (H)
            return context_vector 

class FactorPredictor(nn.Module):
    def __init__(self, batch_size, hidden_size, num_factor):
        super(FactorPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_factor = num_factor
        self.attention_layers = nn.ModuleList([AttentionLayer(self.hidden_size) for _ in range(num_factor)])
        
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()

    def forward(self, stock_latent):
        #! 오직 stock latent만을 입력으로 받음 (N, H)
        
        for i in range(self.num_factor):
            attention_layer = self.attention_layers[i](stock_latent)
            if i == 0:
                h_multi = attention_layer
            else:
                h_multi = torch.cat((h_multi, attention_layer), dim=0)
        h_multi = h_multi.view(self.num_factor, -1)

        # print("h_multi:", h_multi.shape)
        h_multi = self.linear(h_multi)
        h_multi = self.leakyrelu(h_multi)
        pred_mu = self.mu_layer(h_multi)
        pred_sigma = self.sigma_layer(h_multi)
        pred_sigma = self.softplus(pred_sigma)
        pred_mu = pred_mu.view(-1)
        pred_sigma = pred_sigma.view(-1)
        return pred_mu, pred_sigma

class FactorVAE_old(nn.Module):
    def __init__(self, feature_extractor, factor_encoder, factor_decoder, factor_predictor):
        super(FactorVAE, self).__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_predictor = factor_predictor

    @staticmethod
    def KL_Divergence(mu1, sigma1, mu2, sigma2):
        #! mu1, mu2: (batch_size, 1)
        #! sigma1, sigma2: (batch_size, 1)
        #! output: (batch_size, 1)
        kl_div = (torch.log(sigma2/ sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5).sum()
        return kl_div

    def forward(self, x, returns):
        #! x: (batch_size, seq_length, num_latent)
        #! returns: (batch_size, 1)
        stock_latent = self.feature_extractor(x)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        reconstruction = self.factor_decoder(stock_latent, factor_mu, factor_sigma)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)

        # print(f"pred_mu: {pred_mu.shape}, pred_sigma: {pred_sigma.shape}")
        # Define VAE loss function with reconstruction loss and KL divergence
        reconstruction_loss = F.mse_loss(reconstruction, returns)
        # Calculate KL divergence between two Gaussian distributions
        if torch.any(pred_sigma == 0):
            pred_sigma[pred_sigma == 0] = 1e-6
        kl_divergence = self.KL_Divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)

        vae_loss = reconstruction_loss + kl_divergence
        # print("loss: ", vae_loss)
        return vae_loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma #! reconstruction, factor_mu, factor_sigma

    # 학습 이후 사용
    def prediction(self, x):
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        y_pred = self.factor_decoder(stock_latent, pred_mu, pred_sigma)

        return y_pred
    
class FactorVAE(nn.Module):
    def __init__(self, feature_extractor, factor_encoder, factor_decoder, factor_predictor):
        super(FactorVAE, self).__init__()
        self.feature_extractor = feature_extractor
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
        self.factor_predictor = factor_predictor

    @staticmethod
    def KL_Divergence(mu1, sigma1, mu2, sigma2):
        #! mu1, mu2: (batch_size, 1)
        #! sigma1, sigma2: (batch_size, 1)
        #! output: (batch_size, 1)
        kl_div = (torch.log(sigma2/ sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5).sum()
        return kl_div

    def forward(self, x, returns):
        #! x: (batch_size, seq_length, num_latent)
        #! returns: (batch_size, 1)

        stock_latent = self.feature_extractor(x)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        reconstruction = self.factor_decoder(stock_latent, factor_mu, factor_sigma)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)

        # print(f"pred_mu: {pred_mu.shape}, pred_sigma: {pred_sigma.shape}")
        # Define VAE loss function with reconstruction loss and KL divergence
        #* Some adjustment
        #* stock_adj: number of stocks that have no return data
        stock_adj = 0
        for i in range(len(returns)-1,-1,-1):
            if returns[i] == 0:
                stock_adj += 1
            else:
                break

        if stock_adj > 0:
            reconstruction_loss = F.mse_loss(reconstruction[:-stock_adj], returns[:-stock_adj])
        else:
            reconstruction_loss = F.mse_loss(reconstruction, returns)
            
        # Calculate KL divergence between two Gaussian distributions
        if torch.any(pred_sigma == 0):
            pred_sigma[pred_sigma == 0] = 1e-6
        kl_divergence = self.KL_Divergence(factor_mu, factor_sigma, pred_mu, pred_sigma)

        vae_loss = reconstruction_loss + kl_divergence
        # print("loss: ", vae_loss)
        return vae_loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma #! reconstruction, factor_mu, factor_sigma

    # 학습 이후 사용
    def prediction(self, x):
        stock_latent = self.feature_extractor(x)
        pred_mu, pred_sigma = self.factor_predictor(stock_latent)
        y_pred = self.factor_decoder(stock_latent, pred_mu, pred_sigma)

        return y_pred

#%%    
num_latent = 20
batch_size = 300 # equal to num of stocks
seq_len = 30
num_factor = 8
hidden_size = 20

test_char = torch.randn(batch_size, seq_len, num_latent) # (batch_size, seq_length, num_latent)
test_returns = torch.randn(batch_size, 1) # (batch_size, 1)

feature_extractor = FeatureExtractor(num_latent = num_latent, hidden_size =hidden_size)
stock_latent = feature_extractor(test_char)

factor_encoder = FactorEncoder(num_factors=num_factor, num_portfolio=num_latent, hidden_size=hidden_size)
alpha_layer = AlphaLayer(hidden_size)
beta_layer = BetaLayer(hidden_size, num_factor)
factor_decoder = FactorDecoder(alpha_layer, beta_layer)
factor_predictor = FactorPredictor(batch_size, hidden_size, num_factor)
factorVAE = FactorVAE(feature_extractor, factor_encoder, factor_decoder, factor_predictor)

vae_loss, reconstruction, factor_mu, factor_sigma, pred_mu, pred_sigma = factorVAE(test_char, test_returns)

print(vae_loss, factor_mu, factor_sigma, pred_mu, pred_sigma)
