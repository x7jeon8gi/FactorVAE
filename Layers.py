import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# GRU을 통한 feature extraction, 입력으로 주식들의 firm characteristic을 받아서, firm characteristic을 통해 주식의 latent vector를 추출
class FeatureExtractor(nn.Module):
    def __init__(self, num_latent, hidden_size, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.num_latent = num_latent
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.linear = nn.Linear(num_latent, num_latent)
        self.leakyrelu = nn.LeakyReLU()
        self.gru = nn.GRU(num_latent, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        #! x: (batch_size, seq_length, num_latent)
        # Apply linear and LeakyReLU activation
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
        weights = self.softmax(weights)

        # multiply weights and returns
        print(f"weights shape: {weights.shape}, returns shape: {returns.shape}") # [256, 20], [256, 1]
        portfolio_return = torch.mm(weights.transpose(1,0), returns) #* portfolio_return: (M, 1)
        print(f"portfolio_return shape: {portfolio_return.shape}")
        
        return self.mapping_layer(portfolio_return)

class AlphaLayer(nn.Module):
    def __init__(self, hidden_size, num_factors):
        super(AlphaLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, num_factors)
        self.sigma_layer = nn.Linear(hidden_size, num_factors)
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
        self.linear1 = nn.Linear(hidden_size, num_factor)
    
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
        alpha_mu, alpha_sigma = self.alpha_layer(stock_latent)
        print(f"alpha_mu shape: {alpha_mu.shape}, alpha_sigma shape: {alpha_sigma.shape}")
        beta = self.beta_layer(stock_latent)

        factor_mu = factor_mu.view(-1, 1)
        factor_sigma = factor_sigma.view(-1, 1)
        print(f"factor_mu shape: {factor_mu.shape}, factor_sigma shape: {factor_sigma.shape}")
        print(f"beta shape: {beta.shape}")
        mu = alpha_mu + torch.matmul(beta, factor_mu)
        sigma = torch.sqrt(alpha_sigma**2 + torch.matmul(beta**2, factor_sigma**2))

        return self.reparameterize(mu, sigma)
         
class FactorVAE(nn.Module):
    def __init__(self, factor_encoder, factor_decoder):
        super(FactorVAE, self).__init__()
        self.factor_encoder = factor_encoder
        self.factor_decoder = factor_decoder
    
    def forward(self, stock_latent, returns):
        #! x: (batch_size, seq_length, num_latent)
        #! returns: (batch_size, 1)
        factor_mu, factor_sigma = self.factor_encoder(stock_latent, returns)
        return self.factor_decoder(stock_latent, factor_mu, factor_sigma), factor_mu, factor_sigma #! reconstruction, factor_mu, factor_sigma


#todo attention layer 효율성 개선 필요: for loop 제거해야됨
class FactorPredictor(nn.Module):
    def __init__(self, batch_size, hidden_size, num_factor):
        super(FactorPredictor, self).__init__()

        # self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)
        
        self.query = nn.Parameter(torch.randn(hidden_size)) #.repeat(batch_size, 1)
        self.dropout = nn.Dropout(0.1)
        self.num_factor = num_factor
        
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.leakyrelu = nn.LeakyReLU()
        self.mu_layer = nn.Linear(hidden_size, 1)
        self.sigma_layer = nn.Linear(hidden_size, 1)
        self.softplus = nn.Softplus()
        
    def calculate_attention(self, query, key, value):
        #! query: (H), key: (N, H), value: (N, H) #* query를 제대로 정의하지 못하여 헷갈림
        #! output: (K, H)
        
        #* calculate attention weights
        attention_weights = torch.matmul(query, key.transpose(1,0)) # (N)
        #* scaling
        attention_weights = attention_weights / torch.sqrt(torch.tensor(key.shape[0]))
        # print(f"attention_weights shape: {attention_weights.shape}")
        attention_weights = self.dropout(attention_weights)
        attention_weights = F.relu(attention_weights) # max(0, x)
        attention_weights = F.softmax(attention_weights, dim=0) # (N)
        
        #! calculate context vector
        #* 이걸 K개 만큼 쌓아서 올리면 K*H dimension이 나올 것
        context_vector = torch.matmul(attention_weights, value) # (H)
        
        return context_vector # , attention_weights
    
    def distribution_network(self, h_multi):
        #todo dimension 일치 필요 
        # h_multi: (num_factor, H)
        
        h_multi = self.linear(h_multi) # (num_factor, H)
        h_multi = self.leakyrelu(h_multi) 
        mu = self.mu_layer(h_multi) # (num_factor, 1)
        sigma = self.sigma_layer(h_multi) # (num_factor, 1)
        sigma = self.softplus(sigma)        
    
    def forward(self, stock_latent):
        #! 오직 stock latent만을 입력으로 받음 (N, H)
        
        self.key = self.key_layer(stock_latent)
        self.value = self.value_layer(stock_latent)
        
        h_multi = torch.stack([self.calculate_attention(self.query, self.key, self.value) for _ in range(self.num_factor)], dim=0)
        # (num_factor, H)
        
        h_multi = self.linear(h_multi)
        h_multi = self.leakyrelu(h_multi)
        pred_mu = self.mu_layer(h_multi)
        pred_sigma = self.sigma_layer(h_multi)
        pred_sigma = self.softplus(pred_sigma)
        return pred_mu, pred_sigma
