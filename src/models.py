import torch
import torch.nn as nn

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        """
        初始化RBM模型

        :param n_visible: 可見層模型
        :param n_hidden: 隱藏層模型
        """
        # 初始化權重矩陣，使用小的隨機值
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 0.01)  # 權重矩陣

        # 初始化bias為0
        self.v_bias = nn.Parameter(torch.zeros(n_visible))  # 可見層偏置
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))  # 隱藏層偏置

    def sample_from_p(self, p):
        """
        根據概率分布p進行伯努利採樣
        """
        return torch.bernoulli(p)  # 根據概率分布進行伯努利採樣
    
    def v_to_h(self, v):
        """
        從可見層到隱藏層的轉換
        性質就有點像是模型看到一張圖片，然後將visible的東西計入隱藏層的激活概率中，然後根據這個概率來決定哪些隱藏層的神經元被激活了

        :param v: 可見層的輸入
        """
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)  # 計算隱藏層的激活概率
        return p_h, self.sample_from_p(p_h)  # 返回激活概率和採樣結果
    
    def h_to_v(self, h):
        """
        從隱藏層到可見層的轉換

        :param h: 隱藏層的輸入
        """
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)  # 計算可見層的激活概率
        return p_v, self.sample_from_p(p_v)  # 返回激活概率和採樣結果
    
    def contrastive_divergence(self, v, lr=0.1, k=1):
        """
        使用對比散度算法進行參數更新

        :param v: 可見層的輸入 (batch_size, n_visible)
        :param k: CD-k中的k值，表示採樣的步數
        """
        v0 = v  # 初始可見層
        ph0, h0 = self.v_to_h(v0)  # 初始隱藏層

        # Gibbs採樣過程
        vk = v0
        for _ in range(k):
            _, hk = self.v_to_h(vk)  # 從可見層到隱藏層
            _, vk = self.h_to_v(hk)  # 從隱藏層到可見層

        phk, _ = self.v_to_h(vk)  # 最終隱藏層

        # 計算梯度並更新參數
        w_grad = torch.matmul(v0.t(), ph0) - torch.matmul(vk.t(), phk)  # 權重梯度
        v_bias_grad = torch.sum(v0 - vk, dim=0)  # 可見層偏置梯度
        h_bias_grad = torch.sum(ph0 - phk, dim=0)  # 隱藏層偏置梯度

        # 使用無梯度上下文來更新參數，避免PyTorch的自動求導機制干擾
        with torch.no_grad():
            self.W += lr * w_grad / v.size(0)  # 更新權重
            self.v_bias += lr * v_bias_grad / v.size(0)  # 更新可見層偏置
            self.h_bias += lr * h_bias_grad / v.size(0)  # 更新隱藏層偏置