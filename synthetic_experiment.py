# %%
import json
import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn.functional as F
import torch.nn.init as init

# Constants
d1, d = 2500, 50
m, B = 80, 50 * 10
total_samples, test_samples = 5000, 2000
Sw = total_samples
Sh = 3000
sigma_xi = 8/d

# Generate orthonormal matrices
M, _ = torch.qr(torch.randn(d1, d))
H, _ = torch.qr(torch.randn(d1, d))
M, H = M, H

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
retrain_results = {}
results = {}
mean_elements_above_threshold = {}

# 存储多次实验的结果
all_results = []
all_retrain_results = []

class MulticlassImageTextDataset(Dataset):
    def __init__(self, total_samples, M, d1, d, sigma_xi, num_classes=5):
        self.total_samples = total_samples
        self.M = M
        self.d1, self.d = d1, d
        self.sigma_xi = sigma_xi
        self.num_classes = num_classes
        self.segment_size = d // num_classes
        
        # Define text labels for each class
        self.text_labels = [
            "zero", "one", "two", "three", "four", 
            "five", "six", "seven", "eight", "nine"
        ]
        # self.text_labels = [
        #     "zero", "one", "two", "three", "four", 
        #     "five"
        # ]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        while True:
            # Generate z_x using Bernoulli distribution
            z_x = torch.bernoulli(torch.full((self.d,), 0.1))

            # Determine the class based on which segment has the most 1s
            segments = z_x.view(self.num_classes, -1)
            segment_sums = segments.sum(dim=1)
            
            # Sort segment sums in descending order
            sorted_sums, _ = torch.sort(segment_sums, descending=True)
            
            # Check if the maximum sum is at least 2 more than the second highest
            # and there are no ties for the maximum
            if sorted_sums[0] >= sorted_sums[1] + 2:
                break

        # Now determine the true class (no need to break ties)
        true_class = torch.argmax(segment_sums)

        # Create one-hot encoding for the class
        one_hot = torch.zeros(self.num_classes)
        one_hot[true_class] = 1

        # Generate x
        noise = torch.randn(self.d1) * self.sigma_xi
        x = torch.matmul(self.M, z_x) + noise

        # Get the text label for the class
        y_text = self.text_labels[true_class]

        return x, y_text, z_x, one_hot

def create_dataset(total_samples, M, d1=d1, d=d, sigma_xi=sigma_xi, num_classes=5, batch_size=B, train_ratio=0.8):
    dataset = MulticlassImageTextDataset(total_samples, M, d1, d, sigma_xi, num_classes)
    
    # Split the dataset into train and test
    train_size = int(train_ratio * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

class ImageTextDataset(Dataset):
    def __init__(self, total_samples, M, H, d1, d, sigma_xi, noise_prob=0.5):
        self.total_samples, self.M, self.H = total_samples, M, H
        self.d1, self.d, self.sigma_xi, self.noise_prob = d1, d, sigma_xi, noise_prob

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # Generate z_x
        z_x = torch.bernoulli(torch.full((self.d,), 0.1))
        
        # Determine if z_y should be random or equal to z_x
        if torch.rand(1).item() < self.noise_prob:
            # Generate z_y randomly
            z_y = torch.bernoulli(torch.full((self.d,), 0.1))
        else:
            # Set z_y equal to z_x
            z_y = z_x.clone()
        
        xi_x, xi_y = torch.randn(self.d1) * self.sigma_xi, torch.randn(self.d1) * self.sigma_xi
        x, y = torch.matmul(z_x, self.M.T) + 0*xi_x, torch.matmul(z_y, self.H.T) + xi_y
        return x, y, z_x, z_y

class SequentialSpecializedImageTextDataset(Dataset):
    def __init__(self, M, H, d1, d, sigma_xi):
        self.M, self.H = M, H
        self.d1, self.d, self.sigma_xi = d1, d, sigma_xi
        self.total_samples = d  # 总样本数等于 d

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        if idx >= self.d:
            raise IndexError("Index out of bounds")

        # 创建一个只在 idx 位置有 1 的向量
        z = torch.zeros(self.d)
        z[idx] = 1.0

        # z_x 和 z_y 相同
        z_x = z
        z_y = z

        # 生成 x 和 y
        xi_y = torch.randn(self.d1) * self.sigma_xi
        x = torch.matmul(z_x, self.M.T)  # 注意：这里不添加噪声
        y = torch.matmul(z_y, self.H.T) + xi_y

        return x, y, z_x, z_y

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim, bias=True)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        init.normal_(self.linear1.weight, mean=0.0, std=1e-3)
        if self.linear1.bias is not None:
            # init.zeros_(self.linear1.bias)
            init.normal_(self.linear1.bias, mean=0.0, std=1e-3)
    
    def _normalize_weights_and_bias(self):
        with torch.no_grad():
            norms = torch.norm(self.linear1.weight, p='fro', dim=1)
            norms_unsqueezed = norms.unsqueeze(1)
            self.linear1.weight.div_(norms_unsqueezed)
            if self.linear1.bias is not None:
                self.linear1.bias.div_(norms_unsqueezed.squeeze())

    def forward(self, x):
        self._normalize_weights_and_bias()
        x = self.relu(self.linear1(x))
        return x

def check_matrix_mult_threshold(W_image, M, threshold=0.8):
    # 确保 W_image 和 M 在同一设备上
    W_image = W_image.to(device)
    M = M.to(device)
    
    # 执行矩阵乘法
    result = torch.matmul(W_image, M)
    
    # 检查哪些值大于阈值
    above_threshold = (result > threshold).float()
    
    # 计算每行中大于阈值的元素数量
    count_above_threshold = above_threshold.sum(dim=1)
    
    # 统计信息
    total_above = above_threshold.sum().item()
    mean_above = count_above_threshold.mean().item()
    max_above = count_above_threshold.max().item()
    min_above = count_above_threshold.min().item()
    
    # 计算达到最大值和最小值的行数
    max_count = (count_above_threshold == max_above).sum().item()
    min_count = (count_above_threshold == min_above).sum().item()
    
    # print(f"Total elements > {threshold}: {total_above}")
    # print(f"Mean elements > {threshold} per row: {mean_above:.2f}")
    # print(f"Max elements > {threshold} in a row: {max_above}")
    # print(f"Number of rows with max elements: {max_count}")
    # print(f"Min elements > {threshold} in a row: {min_above}")
    # print(f"Number of rows with min elements: {min_count}")
    
    return above_threshold, count_above_threshold

def info_nce_loss(features, image_encoder=None, text_encoder=None, lambda_w=0.001, lambda_v=0.001, temperature=0.1):
    device = features.device
    batch_size = features.shape[0] // 2
    
    # 创建标签
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0).to(device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().to(device)
    
    # 计算相似性矩阵
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # 创建掩码并移除对角线元素
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    
    # 计算正样本和负样本的相似性
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)
    
    # 拼接 logits 并计算损失
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    loss = F.cross_entropy(logits, labels)
    
    # 计算 Frobenius 正则化项（如果有 encoder）
    if image_encoder is not None:
        W_frobenius_squared = sum(torch.norm(param, p='fro') ** 2 for param in image_encoder.parameters())
        loss += lambda_w * W_frobenius_squared
    
    if text_encoder is not None:
        V_frobenius_squared = sum(torch.norm(param, p='fro') ** 2 for param in text_encoder.parameters())
        loss += lambda_v * V_frobenius_squared
    
    return loss

def evaluate_model(test_dataset, image_encoder, text_encoder):
    device = image_encoder.linear1.weight.device
    x, y, z_x, z_y = map(lambda t: t.to(device), map(torch.stack, zip(*[test_dataset[i] for i in range(len(test_dataset))])))
    
    feature_x, feature_y = image_encoder(x), text_encoder(y)
    feature_x_norm, feature_y_norm = F.normalize(feature_x, p=2, dim=1), F.normalize(feature_y, p=2, dim=1)
    result_features = torch.matmul(feature_x_norm, feature_y_norm.T)
    
    z_x_norm, z_y_norm = F.normalize(z_x.float(), p=2, dim=1), F.normalize(z_y.float(), p=2, dim=1)
    result_z = torch.matmul(z_x_norm, z_y_norm.T)
    
    error_matrix = torch.abs(result_features - result_z)
    
    diag_avg_features = torch.mean(torch.diagonal(result_features)).item()
    non_diag_avg_features = (torch.sum(result_features) - torch.sum(torch.diagonal(result_features))).item() / (result_features.numel() - result_features.size(0))
    mean_error = torch.mean(error_matrix).item()
    diagonal_error = torch.mean(torch.abs(torch.diagonal(result_features) - torch.diagonal(result_z))).item()
    non_diagonal_error = (torch.sum(error_matrix) - torch.sum(torch.diagonal(error_matrix))).item() / (result_features.size(0) ** 2 - result_features.size(0))
    correlation = torch.corrcoef(torch.stack([result_features.flatten(), result_z.flatten()]))[0, 1].item()
    
    results = [diag_avg_features, non_diag_avg_features, mean_error, diagonal_error, non_diagonal_error, correlation]
    
    print(f"Avg diagonal: {results[0]:.4f}")
    print(f"Avg non-diagonal: {results[1]:.4f}")
    # print(f"Mean error: {results[2]:.4f}")
    # print(f"Diagonal error: {results[3]:.4f}")
    # print(f"Non-diagonal error: {results[4]:.4f}")
    # print(f"Correlation: {results[5]:.4f}")
    
    return results

def compare_results(results_dict):
    metrics = ["Avg diagonal", "Avg non-diagonal", "Mean error", "Diagonal error", "Non-diagonal error", "Correlation"]
    noise_probs = list(results_dict.keys())
    
    print("\nComparison of results (all evaluated on clean test data):")
    header = "Metric".ljust(20) + "".join([f"Train Noise {np:.2f}".ljust(20) for np in noise_probs])
    print(header)
    print("-" * len(header))
    
    for i, metric in enumerate(metrics):
        row = metric.ljust(20)
        for np in noise_probs:
            row += f"{results_dict[np][i]:<20.4f}"
        print(row)

def extract_encoder_parameters(encoder):
    """
    从编码器中提取线性层的权重和偏置。
    可用于 image_encoder 或 text_encoder。
    """
    if not hasattr(encoder, 'linear1'):
        raise AttributeError("Encoder does not have a 'linear1' attribute. Check the encoder structure.")
    
    # 提取权重 W
    W = encoder.linear1.weight.data.clone()
    
    # 提取偏置 b
    b = encoder.linear1.bias.data.clone() if encoder.linear1.bias is not None else None
    
    return W, b

class ModifiedSwDataset(Dataset):
    def __init__(self, original_dataset, replaced_samples):
        self.original_dataset = original_dataset
        self.replaced_samples = replaced_samples

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        x, _, z_x, z_y = self.original_dataset[idx]
        x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x
        replaced_y = torch.tensor(self.replaced_samples[idx], dtype=torch.float32)
        z_x = torch.tensor(z_x, dtype=torch.float32) if not torch.is_tensor(z_x) else z_x
        z_y = torch.tensor(z_y, dtype=torch.float32) if not torch.is_tensor(z_y) else z_y

        return x, replaced_y, z_x, z_y

class SmallDatasetImageDecoder(nn.Module):
    def __init__(self, input_dim=2500, hidden_dim=80, output_dim=2500):
        super(SmallDatasetImageDecoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

def generate_and_replace_samples(image_decoder, dataset, batch_size=500):
    image_decoder.eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    replaced_samples = []

    with torch.no_grad():
        for batch_idx, (x_batch, y_batch, z_x_batch, z_y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            
            new_y_batch = image_decoder(x_batch)
            
            replaced_samples.extend(new_y_batch.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f'Processed {batch_idx + 1} batches')

    print(f"Total samples replaced: {len(replaced_samples)}")
    replacement_ratio = len(replaced_samples) / len(dataset)
    
    return ModifiedSwDataset(dataset, replaced_samples), replacement_ratio

def train_decoder(model, train_loader, device, num_epochs=40, learning_rate=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch, _, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            # learning_rate = learning_rate * 0.5
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

def train_and_evaluate(filtered_train_loader, epochs=20, lr=3e-3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_encoder = Encoder(input_dim=d1, output_dim=m).to(device)
    text_encoder = Encoder(input_dim=d1, output_dim=m).to(device)
    optimizer = torch.optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=lr)
    for epoch in range(epochs):
        for x_batch, y_batch, _, _ in filtered_train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            features = torch.cat([image_encoder(x_batch), text_encoder(y_batch)], dim=0)
            loss = info_nce_loss(features, image_encoder, text_encoder)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    test_dataset_clean = ImageTextDataset(total_samples=test_samples, M=M, H=H, d1=d1, d=d, sigma_xi=sigma_xi, noise_prob=0)
    # print(f"\nEvaluation on clean test dataset (trained with filtered dataset):")
    results = evaluate_model(test_dataset_clean, image_encoder, text_encoder)

    return results, image_encoder, text_encoder

def finetune(pre_trained_image_encoder, pre_trained_text_encoder, train_dataset_Sh, test_dataset_clean, epochs=40, lr=1e-3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_encoder = Encoder(input_dim=d1, output_dim=m).to(device)
    text_encoder = Encoder(input_dim=d1, output_dim=m).to(device)
    image_encoder.load_state_dict(pre_trained_image_encoder.state_dict())
    text_encoder.load_state_dict(pre_trained_text_encoder.state_dict())

    optimizer = torch.optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_loader = DataLoader(train_dataset_Sh, batch_size=B, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(epochs):
        image_encoder.train()
        text_encoder.train()
        for x_batch, y_batch, _, _ in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            features = torch.cat([image_encoder(x_batch), text_encoder(y_batch)], dim=0)
            loss = info_nce_loss(features, image_encoder, text_encoder)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    final_eval_results = evaluate_model(test_dataset_clean, image_encoder, text_encoder)
    # print("\nEvaluation on clean test dataset:", final_eval_results)
    return final_eval_results, image_encoder, text_encoder

def analyze_matrix_mult_results(W_image, M, threshold=0.5):
    device = W_image.device
    M = M.to(device)
    
    # Normalize W_image
    norms = torch.norm(W_image, p='fro', dim=1)
    norms_unsqueezed = norms.unsqueeze(1)
    W_image_normalized = W_image.div(norms_unsqueezed)
    
    result = torch.matmul(W_image_normalized, M)
    above_threshold = (torch.abs(result) > threshold).float()
    
    single_position_rows = {}
    total_magnitude = 0
    for i, row in enumerate(above_threshold):
        positions = row.nonzero().squeeze().tolist()
        if isinstance(positions, int):  # If only one position, convert to list
            positions = [positions]
        if len(positions) == 1:
            single_position_rows[i] = positions[0]
            total_magnitude += torch.abs(result[i, positions[0]]).item()

    unique_positions = set(single_position_rows.values())
    unique_count = len(unique_positions)
    
    avg_magnitude = total_magnitude / unique_count if unique_count > 0 else 0

    return single_position_rows, unique_count, avg_magnitude


# class ModifiedSwDataset(Dataset):
    def __init__(self, original_dataset, kept_indices):
        self.original_dataset = original_dataset
        self.kept_indices = kept_indices

    def __len__(self):
        return len(self.kept_indices)

    def __getitem__(self, idx):
        return self.original_dataset[self.kept_indices[idx]]

def optimize_analyze_matrix_mult_results(W_image, M, threshold_start=0.2, threshold_end=1.0, threshold_step=0.05):
    max_unique_count = 0
    optimal_avg_magnitude = 0

    for threshold in torch.arange(threshold_start, threshold_end + threshold_step, threshold_step):
        single_position_rows, unique_count, avg_magnitude = analyze_matrix_mult_results(W_image, M, threshold=threshold.item())

        if unique_count > max_unique_count:
            max_unique_count = unique_count
            optimal_avg_magnitude = avg_magnitude

    return max_unique_count, optimal_avg_magnitude

def generate_new_dataset(image_decoder, image_encoder, text_encoder, original_dataset, batch_size):
    image_decoder.eval()
    
    dataloader = DataLoader(original_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    new_data = []
    replaced_count = 0
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch, z_x_batch, z_y_batch) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Generate new y using the decoder
            new_y_batch = image_decoder(x_batch)
            
            encoded_x = image_encoder(x_batch)
            encoded_y = text_encoder(y_batch)
            encoded_new_y = text_encoder(new_y_batch)
            
            cos_sim_original = F.cosine_similarity(encoded_x, encoded_y, dim=1)
            cos_sim_new = F.cosine_similarity(encoded_x, encoded_new_y, dim=1)
            
            similarity_threshold = 1.2
            # if np < 0.3:
            #     similarity_threshold = 1.2
            # else:
            #     similarity_threshold = 1.2
            
            # Calculate cosine similarity
            # cos_sim_new = F.cosine_similarity(x_batch, new_y_batch)
            # cos_sim_original = F.cosine_similarity(x_batch, y_batch)
            
            # Compare similarities and filter data
            
            for x, y, new_y, z_x, z_y, sim_new, sim_original in zip(x_batch, y_batch, new_y_batch, z_x_batch, z_y_batch, cos_sim_new, cos_sim_original):
                if sim_new >= similarity_threshold * sim_original:
                    new_data.append((x.cpu(), new_y.cpu(), z_x, z_y))
                    replaced_count += 1
                else:
                    new_data.append((x.cpu(), y.cpu(), z_x, z_y))  # Keep original if similarity is too low

            if (batch_idx + 1) % 10 == 0:
                print(f'Processed {batch_idx + 1} batches')

    replacement_ratio = replaced_count / len(original_dataset)
    print(f"Replacement ratio: {replacement_ratio:.2%}")
    print(f"Samples with new y: {sum(1 for _, y, _, _ in new_data if not torch.allclose(y, original_dataset[_][1]))}")
    
    return NewDataset(new_data), replacement_ratio

class NewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y, z_x, z_y = self.data[idx]
        return (
            x.clone().detach() if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32),
            y.clone().detach() if torch.is_tensor(y) else torch.tensor(y, dtype=torch.float32),
            z_x.clone().detach() if torch.is_tensor(z_x) else torch.tensor(z_x, dtype=torch.float32),
            z_y.clone().detach() if torch.is_tensor(z_y) else torch.tensor(z_y, dtype=torch.float32)
        )

def train_decoder(decoder, train_loader, num_epochs=50, learning_rate=1e-3):
    decoder.train()
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for x_batch, y_batch, _, _ in train_loader:
            # 设备检查
            if x_batch.device != device:
                x_batch = x_batch.to(device)
            if y_batch.device != device:
                y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = decoder(x_batch)
            
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
            learning_rate = learning_rate * 0.5

class CustomSmallDatasetImageDecoder(nn.Module):
    def __init__(self, M, H, use_given_matrices=True):
        super(CustomSmallDatasetImageDecoder, self).__init__()
        
        d1, m = M.shape
        assert M.shape == (d1, m), f"M should be of shape ({d1}, {m})"
        assert H.shape == (d1, m), f"H should be of shape ({d1}, {m})"
        
        self.model = nn.Sequential(
            nn.Linear(d1, m, bias=True),
            nn.ReLU(),
            nn.Linear(m, d1, bias=True)
        )
        
        if use_given_matrices:
            # Set the weights of the first and last linear layers
            self.model[0].weight.data = M.t()  # Transpose of M
            self.model[2].weight.data = H
            
            # Initialize biases to zero
            nn.init.zeros_(self.model[0].bias)
            nn.init.zeros_(self.model[2].bias)
        else:
            # Random initialization
            nn.init.xavier_uniform_(self.model[0].weight)
            nn.init.xavier_uniform_(self.model[2].weight)
            nn.init.zeros_(self.model[0].bias)
            nn.init.zeros_(self.model[2].bias)

    def forward(self, x):
        return self.model(x)

class MulticlassImageTextDataset(Dataset):
    def __init__(self, total_samples, M, d1, d, sigma_xi, num_classes):
        self.total_samples = total_samples
        self.M = M
        self.d1, self.d = d1, d
        self.sigma_xi = sigma_xi
        self.num_classes = num_classes
        self.segment_size = d // num_classes
        self.text_labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        while True:
            z_x = torch.bernoulli(torch.full((self.d,), 0.1))
            segments = z_x.view(self.num_classes, -1)
            segment_sums = segments.sum(dim=1)
            sorted_sums, _ = torch.sort(segment_sums, descending=True)
            if sorted_sums[0] >= sorted_sums[1] + 2:
                break

        true_class = torch.argmax(segment_sums)
        one_hot = torch.zeros(self.num_classes)
        one_hot[true_class] = 1

        noise = torch.randn(self.d1) * self.sigma_xi
        x = torch.matmul(self.M, z_x) + noise

        y_text = self.text_labels[true_class]

        return x, y_text, z_x, one_hot

class MulticlassTextDataset(Dataset):
    def __init__(self, H, d1, d, sigma_xi, num_classes):
        self.H, self.d1, self.d = H, d1, d
        self.sigma_xi = sigma_xi
        self.num_classes = num_classes
        self.segment_size = d // num_classes
        self.text_labels = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    def __len__(self):
        return self.num_classes

    def __getitem__(self, idx):
        z_x = torch.zeros(self.d)
        z_x[idx * self.segment_size : (idx + 1) * self.segment_size] = 1
        
        one_hot = torch.zeros(self.num_classes)
        one_hot[idx] = 1

        noise = torch.randn(self.d1) * self.sigma_xi
        x = torch.matmul(self.H, z_x) + noise

        return x, self.text_labels[idx], z_x, one_hot

def cosine_similarity_gpu(a, b):
    a_norm = a / a.norm(dim=1)[:, None]
    b_norm = b / b.norm(dim=1)[:, None]
    return torch.mm(a_norm, b_norm.transpose(0, 1))

def test_image_text_classification(test_loader, text_data, image_encoder, text_encoder, device):
    # print(f"Starting test_image_text_classification. Text data type: {type(text_data)}, length: {len(text_data)}")
    if len(text_data) == 0:
        print("Error: text_data is empty!")
        return 0.0

    correct = 0
    total = 0
    
    text_samples, _, z_y_samples, _ = text_data
    text_samples = torch.stack([torch.tensor(x) for x in text_samples]).to(device)
    z_y_samples = torch.stack([torch.tensor(z) for z in z_y_samples]).to(device)
    
    with torch.no_grad():
        text_features = text_encoder(text_samples)
    
    for x, y_text, z_x, one_hot in test_loader:
        x, z_x, one_hot = x.to(device), z_x.to(device), one_hot.to(device)
        total += x.size(0)
        
        with torch.no_grad():
            image_features = image_encoder(x)
        
        similarities_x = cosine_similarity_gpu(image_features, text_features)
        similarities_z = cosine_similarity_gpu(z_x, z_y_samples)
        
        pred_x = torch.argmax(similarities_x, dim=1)
        pred_z = torch.argmax(similarities_z, dim=1)
        
        correct_predictions = (pred_x == pred_z) & (pred_x == torch.argmax(one_hot, dim=1))
        correct += torch.sum(correct_predictions).item()
    
    accuracy = correct / total
    return accuracy

def create_dataset(total_samples, M, d1, d, sigma_xi, num_classes, batch_size, train_ratio=0.8):
    dataset = MulticlassImageTextDataset(total_samples, M, d1, d, sigma_xi, num_classes)
    
    train_size = int(train_ratio * total_samples)
    test_size = total_samples - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_Text_dataset(H, d1, d, sigma_xi, num_classes):
    # print(f"Creating text dataset with num_classes: {num_classes}")
    dataset = MulticlassTextDataset(H, d1, d, sigma_xi, num_classes)
    all_samples = [dataset[i] for i in range(num_classes)]
    result = list(zip(*all_samples))
    # print(f"Text dataset created. Number of samples: {len(result[0])}")
    return result

def run_single_experiment(np, M, H, d1, d, sigma_xi, Sw, test_samples, B, downstream_total_samples=5000, num_classes=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 创建数据集和数据加载器
    train_dataset_Sw = ImageTextDataset(total_samples=Sw, M=M, H=H, d1=d1, d=d, sigma_xi=1.5*sigma_xi, noise_prob=np)
    train_loader_Sw = DataLoader(train_dataset_Sw, batch_size=B, shuffle=True, num_workers=4, drop_last=True)
    
    test_dataset_clean = ImageTextDataset(total_samples=test_samples, M=M, H=H, d1=d1, d=d, sigma_xi=sigma_xi, noise_prob=0)
    
    train_dataset_Sh = SequentialSpecializedImageTextDataset(M=M, H=H, d1=d1, d=d, sigma_xi=sigma_xi/10)
    train_loader_Sh = DataLoader(train_dataset_Sh, batch_size=d, shuffle=True, num_workers=4, drop_last=True)
    
    downstream_train_loader, downstream_test_loader = create_dataset(downstream_total_samples, M, d1, d, sigma_xi, num_classes, batch_size=B)
    text_data = create_Text_dataset(H, d1, d, sigma_xi, num_classes)
    
    # 1. 预训练和评估模型
    combined_dataset = ConcatDataset([train_dataset_Sw, train_dataset_Sw])
    combined_loader = DataLoader(combined_dataset, batch_size=B, shuffle=True, num_workers=4, drop_last=True)
    eval_results, image_encoder, text_encoder = train_and_evaluate(combined_loader, epochs=30)
    
    # 提取编码器参数并分析
    W_image, _ = extract_encoder_parameters(image_encoder)
    V_text, _ = extract_encoder_parameters(text_encoder)
    above_threshold, count_above_threshold = check_matrix_mult_threshold(W_image, M, threshold=0.8)
    unique_count, avg_magnitude = optimize_analyze_matrix_mult_results(W_image, M)
    
    # 测试分类器准确率
    test_accuracy = test_image_text_classification(downstream_test_loader, text_data, image_encoder, text_encoder, device)
    
    # 保存初始结果
    initial_results = {
        'M': M, 'H': H, 'eval_results': eval_results,
        'image_encoder': image_encoder, 'text_encoder': text_encoder,
        'W_image': W_image, 'V_text': V_text,
        'above_threshold': above_threshold, 'count_above_threshold': count_above_threshold,
        'mean_elements_above_threshold': count_above_threshold.mean().item(),
        'Unique_positions': unique_count, 'Average_magnitude': avg_magnitude,
        'linear_classifier_accuracy': test_accuracy
    }
    
    # 2. 训练解码器
    if unique_count > d/2:
        decoder = CustomSmallDatasetImageDecoder(W_image.T, V_text.T, use_given_matrices=True).to(device)
        train_decoder(decoder, train_loader_Sh, num_epochs=30, learning_rate=1e-3)
    else:
        decoder = CustomSmallDatasetImageDecoder(W_image.T, V_text.T, use_given_matrices=False).to(device)
        train_decoder(decoder, train_loader_Sh, num_epochs=30, learning_rate=1e-3)
    
    # 3. 生成和替换样本
    filtered_dataset_Sw, replacement_ratio = generate_new_dataset(decoder, image_encoder, text_encoder, train_dataset_Sw, batch_size=B)
    
    if (replacement_ratio < 0.01) and (np < 0.5):
        return initial_results, initial_results  # 在这种情况下，retrain_results 与 initial_results 相同
    
    # 4. 重新训练和评估编码器
    if unique_count > d/2:
        filtered_combined_dataset = ConcatDataset([filtered_dataset_Sw, train_dataset_Sw])
        filtered_combined_loader = DataLoader(filtered_combined_dataset, batch_size=B, shuffle=True, num_workers=4, drop_last=True)
        evaluation_results, new_image_encoder, new_text_encoder = train_and_evaluate(filtered_combined_loader, epochs=30)
    else:
        filtered_combined_dataset = ConcatDataset([filtered_dataset_Sw])
        filtered_combined_loader = DataLoader(filtered_combined_dataset, batch_size=B, shuffle=True, num_workers=4, drop_last=True)
        evaluation_results, new_image_encoder, new_text_encoder = train_and_evaluate(filtered_combined_loader, epochs=60)
    
    # 5. 保存重新训练的结果
    W_image, _ = extract_encoder_parameters(new_image_encoder)
    unique_count, avg_magnitude = optimize_analyze_matrix_mult_results(W_image, M)
    above_threshold, count_above_threshold = check_matrix_mult_threshold(W_image, M, threshold=0.7)
    test_accuracy = test_image_text_classification(downstream_test_loader, text_data, new_image_encoder, new_text_encoder, device)
    
    retrain_results = {
        'M': M, 'H': H, 'eval_results': evaluation_results,
        'image_encoder': new_image_encoder, 'text_encoder': new_text_encoder,'decoder': decoder,
        'W_image': W_image, 'V_text': V_text,
        'above_threshold': above_threshold, 'count_above_threshold': count_above_threshold,
        'mean_elements_above_threshold': count_above_threshold.mean().item(),
        'Unique_positions': unique_count, 'Average_magnitude': avg_magnitude,
        'linear_classifier_accuracy': test_accuracy
    }
    
    return initial_results, retrain_results



# %%
import os
import sys
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def setup_process(rank, world_size, backend='nccl'):
    """初始化分布式进程"""
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group(backend=backend,
                              init_method='env://',
                              world_size=world_size,
                              rank=rank)
        torch.cuda.set_device(rank)
        logging.info(f"Process {rank} connected to process group")
    except Exception as e:
        logging.error(f"Process {rank} failed to initialize: {str(e)}")
        raise e

def cleanup():
    """清理分布式进程"""
    if dist.is_initialized():
        dist.destroy_process_group()

def train(rank, world_size):
    """训练函数"""
    try:
        # 设置进程
        setup_process(rank, world_size)
        
        # 设置数据加载
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        dataset = CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )
        
        loader = DataLoader(
            dataset,
            batch_size=64,
            sampler=sampler,
            num_workers=2,
            pin_memory=True
        )
        
        # 创建模型
        model = SimpleNet().to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        
        # 定义优化器和损失函数
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # 训练循环
        for epoch in range(2):  # 减少epoch数用于测试
            model.train()
            sampler.set_epoch(epoch)
            
            for i, (images, labels) in enumerate(loader):
                images = images.to(rank)
                labels = labels.to(rank)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                if i % 100 == 0 and rank == 0:
                    logging.info(f'Epoch [{epoch+1}/2], Step [{i+1}/{len(loader)}], Loss: {loss.item():.4f}')
            
            # 同步进程
            dist.barrier()
            
        if rank == 0:
            logging.info("Training completed successfully")
    
    except Exception as e:
        logging.error(f"Error in rank {rank}: {str(e)}")
        raise e
    
    finally:
        cleanup()

def main():
    """主函数"""
    try:
        world_size = torch.cuda.device_count()
        if world_size < 2:
            logging.error("Need at least 2 GPUs to run, but got {world_size}")
            return
        
        logging.info(f"Starting training with {world_size} GPUs")
        
        mp.spawn(
            train,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
        
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        # 设置随机种子
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        
        # 设置多进程启动方法
        mp.set_start_method('spawn', force=True)
        
        main()
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        sys.exit(1)

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# 检查是否有多个 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 准备数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# 定义模型（以 ResNet 为例）
model = models.resnet18(pretrained=False)
model = nn.DataParallel(model)  # 使用 DataParallel 包装模型
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(10):  # 假设训练 10 个 epoch
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")
    os.system("nvidia-smi")

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%

# 设置保存路径
save_dir = "/home/sunj11/Documents/VLM/ALBEF/"
os.makedirs(save_dir, exist_ok=True)

# 设置日志
log_file = os.path.join(save_dir, 'experiment.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 将 print 重定向到日志
def print_to_log(*args, **kwargs):
    logging.info(" ".join(map(str, args)), **kwargs)

print = print_to_log

def save_results(results, retrain_results, filename):
    data_to_save = {
        'results': results,
        'retrain_results': retrain_results
    }
    torch.save(data_to_save, filename)
    

# 主程序
if __name__ == "__main__":
    # 设置实验参数
    # 主循环


    train_noise_probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    num_experiments = 15

    for np in train_noise_probs:
        print(f"\nProcessing noise probability: {np}")
        results[np] = {}
        retrain_results[np] = {}
        
        for experiment in range(num_experiments):
            
            print(f"\nRunning experiment {experiment + 1} out of {num_experiments}")
            
            initial_result, retrain_result = run_single_experiment(
                np=np, M=M, H=H, d1=d1, d=d, sigma_xi=sigma_xi, 
                Sw=Sw, test_samples=test_samples, B=B
            )
            
            results[np][experiment] = initial_result
            retrain_results[np][experiment] = retrain_result
            
            print(f"Initial - Noise prob {np}: Unique positions: {initial_result['Unique_positions']}, Average magnitude: {initial_result['Average_magnitude']:.4f}")
            print(f"Initial - Test Accuracy: {initial_result['linear_classifier_accuracy']:.4f}")
            print(f"Retrained - Noise prob {np}: Unique positions: {retrain_result['Unique_positions']}, Average magnitude: {retrain_result['Average_magnitude']:.4f}")
            print(f"Retrained - Test Accuracy: {retrain_result['linear_classifier_accuracy']:.4f}")

    
    # 最终保存所有结果
    # 使用方法
    save_results(results, retrain_results, 'final_experiment_results.pt')

    # 删除检查点文件
    checkpoint_file = os.path.join(save_dir, 'experiment_checkpoint.json')
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed.")


