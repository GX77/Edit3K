import torch

def fetch_indices(tensor, indices):
    return torch.Tensor([tensor[i] for i, value in enumerate(indices) if value == 1])

def top_accuracy(similarity_matrix, categories, type_list):
    n, m = similarity_matrix.shape
    categories_tensor = torch.tensor(categories)

    # 获取每行的前 min(m, 5) 个最高相似度值的索引
    _, sorted_indices = torch.sort(similarity_matrix, descending=True)

    # 计算 Top-1 分类准确率
    top_1_preds = sorted_indices[:, 0]
    top_1_correct = (fetch_indices(top_1_preds, type_list) == fetch_indices(categories_tensor, type_list)).sum().item()
    top_1_accuracy = top_1_correct / sum(type_list)

    # 计算 Top-10 分类准确率（条件允许时）
    k = min(m, 10)  # 获取 min(m, 10) 个最相似的类别
    top_k_preds = sorted_indices[:, :k]
    top_k_correct = (torch.stack([top_k_preds[i] for i in range(len(type_list)) if type_list[i] == 1]) == torch.stack([categories_tensor.view(-1, 1)[i] for i in range(len(type_list)) if type_list[i] == 1])).sum().item()
    top_10_accuracy = top_k_correct / sum(type_list)

    return top_1_accuracy, top_10_accuracy