from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerDecoderLayer

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, in_channels, embed_dim, num_layers, heads, dropout):
        super().__init__()
        assert img_size % patch_size == 0, 'Image dimensions must be divisible by patch size.'
        num_patches = (img_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, embed_dim))  # Adjusted size here
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.ModuleList([
            TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim, dropout=dropout)
            for _ in range(num_layers)])
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.dropout(x)
        for layer in self.transformer:
            x = layer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x