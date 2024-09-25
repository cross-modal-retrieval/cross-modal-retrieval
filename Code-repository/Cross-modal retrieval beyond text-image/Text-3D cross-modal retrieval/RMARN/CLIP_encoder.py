import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel, BertModel, BertTokenizer
import torch.nn.functional as F

def encode_text_single(text):
    # Initialize CLIP model and processor
    local_dir = "/your_path/to/models"
    model = CLIPModel.from_pretrained(local_dir)
    processor = CLIPProcessor.from_pretrained(local_dir)

    # Process text
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)

    # Get text hidden states, keeping the seq_len dimension
    with torch.no_grad():
        outputs = model.text_model(**inputs, output_hidden_states=True, return_dict=True)
        text_embeddings = outputs.hidden_states[-1]  # Get the last layer hidden states

    # Ensure seq_len is 77 by padding or trimming
    text_embeddings = pad_or_trim(text_embeddings, max_len=77)

    # Output tensor shape: (batch_size, seq_len, hidden_size)
    return text_embeddings


def download_bert():
    model_name = "bert-base-chinese"  # Replace with other BERT variants if needed
    # Download BERT model
    model = BertModel.from_pretrained(model_name)
    # Download BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Save model and tokenizer locally
    model_save_path = "/your_path/to/txt_encoder/Bert"  # Replace with desired save path
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"success")

def encode_text_single_with_bert(text):
    # Initialize BERT model and tokenizer
    local_dir = "/your_path/to/txt_encoder/Bert"  # Replace with your BERT model path
    model = BertModel.from_pretrained(local_dir)
    tokenizer = BertTokenizer.from_pretrained(local_dir)

    # Process text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77)

    # Get text hidden states, keeping the seq_len dimension
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        text_embeddings = outputs.hidden_states[-1]  # Get the last layer hidden states

    # Ensure seq_len is 77 by padding or trimming
    text_embeddings = pad_or_trim(text_embeddings, max_len=77)

    # Add downsampling layer, reducing tokens from 77 to 10
    text_embeddings = downsample(text_embeddings, target_len=5)
    text_embeddings = text_embeddings[:, :, :512]  # Retain only the first 512 dimensions
    # Output tensor shape: (batch_size, 10, hidden_size)
    return text_embeddings

def pad_or_trim(tensor, max_len):
    """
    Pad or trim the tensor to have the same sequence length `max_len`.
    """
    batch_size, seq_len, hidden_size = tensor.size()
    if seq_len < max_len:
        # Padding
        pad_size = max_len - seq_len
        padding = torch.zeros(batch_size, pad_size, hidden_size, device=tensor.device)
        return torch.cat((tensor, padding), dim=1)
    elif seq_len > max_len:
        # Trimming
        return tensor[:, :max_len, :]
    else:
        return tensor


def downsample(tensor, target_len):
    # Downsample sequence length from 77 to target_len (10) using average pooling
    batch_size, seq_len, hidden_size = tensor.size()
    tensor = tensor.permute(0, 2, 1)  # Adjust dimensions to (batch_size, hidden_size, seq_len)

    # Kernel size and stride for pooling determine the final output length
    kernel_size = seq_len // target_len  # Calculate kernel size
    stride = kernel_size  # Pooling stride is usually equal to kernel size

    pooled_tensor = F.avg_pool1d(tensor, kernel_size=kernel_size, stride=stride)

    pooled_tensor = pooled_tensor.permute(0, 2, 1)  # Restore dimensions to (batch_size, target_len, hidden_size)

    return pooled_tensor

def encode_text(text_list, max_len=77):
    # Initialize CLIP model and processor
    local_dir = "/your_path/to/models"
    model = CLIPModel.from_pretrained(local_dir)
    processor = CLIPProcessor.from_pretrained(local_dir)

    # Get text encoder from CLIP model
    text_model = model.text_model
    text_features = []

    # Process text
    for texts in text_list:
        embedding_list = []
        for text in texts:
            inputs = processor(text=[text], return_tensors="pt", padding="max_length", truncation=True, max_length=max_len)

            # Get token features from text (model might be running on GPU)
            with torch.no_grad():
                outputs = text_model(**inputs, output_hidden_states=True)
                # Get the last layer token embeddings
                text_embeddings = outputs.last_hidden_state.squeeze(0)[:max_len]

            # print(f"text_embeddings.shape: {text_embeddings.shape}")  # Expected shape: (seq_len, embed_dim)
            embedding_list.append(text_embeddings)

        text_features.append(torch.stack(embedding_list))

    return text_features


if __name__ == '__main__':
    # text_list = [["A beautiful cat.", "A cute dog."], ["A big house."]]
    # text_features = encode_text(text_list)
    # # print(len(text_features))
    # # Check output shape
    # for i, feats in enumerate(text_features):
    #     print(f"text_features[{i}].shape: {feats.shape}")
    # download_bert()
    embeddings = encode_text_single_with_bert("A beautiful cat.")
    print(embeddings.shape)


def encode_textlist(text_list):
    # Initialize CLIP model and processor
    local_dir = "/your_path/to/models"
    model = CLIPModel.from_pretrained(local_dir)
    processor = CLIPProcessor.from_pretrained(local_dir)
    text_features = None
    # Process text
    for _ in text_list:
        for text in _:
            # print(text)
            inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)

            # Get text encoding
            text_embeddings = model.get_text_features(**inputs)
            if text_features is None:
                text_features = text_embeddings
            else:
                text_features = torch.cat((text_features, text_embeddings), dim=0)
    # fc = nn.Linear(text_embeddings.shape[1], 1024)
    # text_embeddings = fc(text_embeddings)
    # print(f'text_features.shape: {text_features.shape}')
    return text_features


def save_clip_model():
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    model.save_pretrained("/your_path/to/models")
    processor.save_pretrained("/your_path/to/models")
